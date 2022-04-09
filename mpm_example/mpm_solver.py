import taichi as ti
import numpy as np
from mpm_example.const import const
from mpm_example.grid_mark import GridMark
from mpm_example.field import Field


@ti.data_oriented
class MPMSolver:
    def __init__(self,
                 max_particle_num,
                 grid_num,
                 ):
        self.max_particle_num = max_particle_num
        self.grid_num = grid_num
        self.dx = 1 / self.grid_num
        self.inv_dx = float(self.grid_num)
        self.dt = 1e-4
        self.steps = 32
        self.p_vol = (self.dx * 0.5) ** 2
        self.bound = 3
        self.E = 5e3
        self.nu = 0.2
        self.mu_0 = self.E / (2 * (1 + self.nu))
        self.lambda_0 = self.E * self.nu / ((1 + self.nu) * (1 - 2 * self.nu))

        self.particles = ti.Struct.field({
            "position": ti.types.vector(3, ti.f32),
            "velocity": ti.types.vector(3, ti.f32),
            "F": ti.types.matrix(3, 3, ti.f32),
            "C": ti.types.matrix(3, 3, ti.f32),
            "Jp": ti.f32,
            "mass": ti.f32,
            "material": ti.i32,
            "color": ti.types.vector(3, ti.f32)
        }, shape=self.max_particle_num)

        self.fluid_node = ti.Struct.field({
            "fluid_m": ti.f32,
            "fluid_v": ti.types.vector(3, ti.f32),
        }, shape=(self.grid_num,) * 3)

        self.solid_node = ti.Struct.field({
            "solid_m": ti.f32,
            "solid_v": ti.types.vector(3, ti.f32),
        }, shape=(self.grid_num,) * 3)

        self.neighbour = (3,) * 3
        self.create_particle_num = ti.field(ti.i32, shape=())
        self.grid_mark = GridMark(self.grid_num)
        self.fluid_field = Field(self.grid_num)
        self.solid_field = Field(self.grid_num)
        self._bear_points = np.load('bear_less.npy')
        self.bear_points = ti.Vector.field(3, dtype=ti.f32, shape=len(self._bear_points))

    def init_bear(self):
        self.bear_points.from_numpy(self._bear_points)

    @ti.kernel
    def reset_node(self):
        for I in ti.grouped(self.fluid_node):
            self.fluid_node[I].fluid_v = ti.zero(self.fluid_node[I].fluid_v)
            self.fluid_node[I].fluid_m = 0
        for I in ti.grouped(self.solid_node):
            self.solid_node[I].solid_v = ti.zero(self.solid_node[I].solid_v)
            self.solid_node[I].solid_m = 0

    @ti.kernel
    def P2G(self):
        for p in range(self.create_particle_num[None]):
            Xp = self.particles[p].position / self.dx
            base = int(Xp - 0.5)
            fx = Xp - base
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            self.particles[p].F = (ti.Matrix.identity(float, 3) + self.dt * self.particles[p].C) @ self.particles[p].F

            h = ti.exp(10 * (1.0 - self.particles[p].Jp))  # Hardening coefficient: snow gets harder when compressed
            if self.particles[p].material == const.MATERIAL_SOLID:  # jelly, make it softer
                h = 0.3
            mu, la = self.mu_0 * h, self.lambda_0 * h
            if self.particles[p].material == const.MATERIAL_FLUID:  # liquid
                mu = 0.0
            U, sig, V = ti.svd(self.particles[p].F)
            J = 1.0
            for d in ti.static(range(3)):
                new_sig = sig[d, d]
                if self.particles[p].material == const.MATERIAL_SNOW:  # Snow
                    new_sig = min(max(sig[d, d], 1 - 2.5e-2),
                                  1 + 4.5e-3)  # Plasticity
                self.particles[p].Jp *= sig[d, d] / new_sig
                sig[d, d] = new_sig
                J *= new_sig
            if self.particles[p].material == const.MATERIAL_FLUID:
                new_F = ti.Matrix.identity(float, 3)
                new_F[0, 0] = J
                self.particles[p].F = new_F
            elif self.particles[p].material == const.MATERIAL_SNOW:
                self.particles[
                    p].F = U @ sig @ V.transpose()  # Reconstruct elastic deformation gradient after plasticity
            stress = 2 * mu * (self.particles[p].F - U @ V.transpose()) @ self.particles[p].F.transpose(
            ) + ti.Matrix.identity(float, 3) * la * J * (J - 1)
            stress = (-self.dt * self.p_vol * 4) * stress / self.dx ** 2
            affine = stress + self.particles[p].mass * self.particles[p].C
            if self.particles[p].material == const.MATERIAL_FLUID:
                for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                    dpos = (offset - fx) * self.dx
                    weight = 1.0
                    for i in ti.static(range(3)):
                        weight *= w[offset[i]][i]
                    self.fluid_node[base + offset].fluid_v += weight * (
                            self.particles[p].mass * self.particles[p].velocity + affine @ dpos)
                    self.fluid_node[base + offset].fluid_m += weight * self.particles[p].mass
            if self.particles[p].material == const.MATERIAL_SOLID:
                for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                    dpos = (offset - fx) * self.dx
                    weight = 1.0
                    for i in ti.static(range(3)):
                        weight *= w[offset[i]][i]
                    self.solid_node[base + offset].solid_v += weight * (
                            self.particles[p].mass * self.particles[p].velocity + affine @ dpos)
                    self.solid_node[base + offset].solid_m += weight * self.particles[p].mass

    @ti.kernel
    def grid_operator(self):
        for I in ti.grouped(self.fluid_node):
            if self.fluid_node[I].fluid_m > 0:
                self.fluid_node[I].fluid_v /= self.fluid_node[I].fluid_m
            self.fluid_node[I].fluid_v += self.dt * ti.Vector([0.0, -9.8, 0.0])
            cond = I < self.bound and self.fluid_node[I].fluid_v < 0 or I > self.grid_num - self.bound and \
                   self.fluid_node[
                       I].fluid_v > 0
            self.fluid_node[I].fluid_v = 0 if cond else self.fluid_node[I].fluid_v
        for I in ti.grouped(self.solid_node):
            if self.solid_node[I].solid_m > 0:
                self.solid_node[I].solid_v /= self.solid_node[I].solid_m
            self.solid_node[I].solid_v += self.dt * ti.Vector([0.0, -9.8, 0.0])
            cond = I < self.bound and self.solid_node[I].solid_v < 0 or I > self.grid_num - self.bound and \
                   self.solid_node[
                       I].solid_v > 0
            self.solid_node[I].solid_v = 0 if cond else self.solid_node[I].solid_v

    @ti.kernel
    def fluidG2P(self):
        for p in range(self.create_particle_num[None]):
            if self.particles[p].material == const.MATERIAL_FLUID:
                Xp = self.particles[p].position / self.dx
                base = int(Xp - 0.5)
                fx = Xp - base
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
                new_v = ti.zero(self.particles[p].velocity)
                new_C = ti.zero(self.particles[p].C)
                for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                    dpos = (offset - fx) * self.dx
                    weight = 1.0
                    for i in ti.static(range(3)):
                        weight *= w[offset[i]][i]
                    g_v = self.fluid_node[base + offset].fluid_v
                    new_v += weight * g_v
                    new_C += 4 * weight * g_v.outer_product(dpos) / self.dx ** 2

                self.particles[p].velocity = new_v
                self.particles[p].position += self.dt * self.particles[p].velocity

                self.particles[p].C = new_C

    @ti.kernel
    def solidG2P(self):
        for p in range(self.create_particle_num[None]):
            if self.particles[p].material == const.MATERIAL_SOLID:
                Xp = self.particles[p].position / self.dx
                base = int(Xp - 0.5)
                fx = Xp - base
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
                new_v = ti.zero(self.particles[p].velocity)
                new_C = ti.zero(self.particles[p].C)
                for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                    dpos = (offset - fx) * self.dx
                    weight = 1.0
                    for i in ti.static(range(3)):
                        weight *= w[offset[i]][i]
                    g_v = self.solid_node[base + offset].solid_v
                    new_v += weight * g_v
                    new_C += 4 * weight * g_v.outer_product(dpos) / self.dx ** 2

                self.particles[p].velocity = new_v
                self.particles[p].position += self.dt * self.particles[p].velocity

                self.particles[p].C = new_C

    @ti.kernel
    def generate_bear(self, material: int):
        rho = 7
        for i in range(len(self._bear_points)):
            n = ti.atomic_add(self.create_particle_num[None], 1)
            zoom = self.bear_points[i] / 3
            position = ti.Vector([zoom.x + 0.3, zoom.y * 0.8 + 0.1, zoom.z + 0.3])
            self.particles[n].position = position
            self.particles[n].material = material
            self.particles[n].velocity = ti.Vector([0.0, 0.0, 0.0])
            self.particles[n].F = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            self.particles[n].mass = self.p_vol * rho
            self.particles[n].Jp = 1
            self.particles[n].color = [1.0, 0.0, 0.0]

    @ti.kernel
    def add_cube(self, position: ti.template(), length: float, particle_num: int, material: int):
        rho = 1
        if material == const.MATERIAL_SOLID:
            rho = 7
        for i in range(self.create_particle_num[None], self.create_particle_num[None] + particle_num):
            n = ti.atomic_add(self.create_particle_num[None], 1)
            self.particles[n].position = ti.Vector([ti.random() for i in range(3)]) * ti.Vector(
                [length, length, length]) + ti.Vector([position[0], position[1], position[2]])
            self.particles[n].material = material
            self.particles[n].velocity = ti.Vector([0.0, 0.0, 0.0])
            self.particles[n].F = ti.Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            self.particles[n].mass = self.p_vol * rho
            self.particles[n].Jp = 1
            if material == const.MATERIAL_SOLID:
                self.particles[n].color = [1.0, 0.0, 0.0]
            if material == const.MATERIAL_FLUID:
                self.particles[n].color = [0.0, 1.0, 1.0]

    def update_field(self):
        self.fluid_field.init()
        self.fluid_field.update(self.fluid_node.fluid_m)
        self.fluid_field.calculate_gradient()

        self.solid_field.init()
        self.solid_field.update(self.solid_node.solid_m)
        self.solid_field.calculate_gradient()

    def run(self, write_ply, frame):
        for s in range(32):
            self.reset_node()
            self.P2G()
            self.update_field()
            self.grid_operator()
            self.fluidG2P()
            self.solidG2P()
        if write_ply:
            print(frame)
            pos = self.particles.position.to_numpy()[:self.create_particle_num[None]]
            np_material = self.particles.material.to_numpy()[:self.create_particle_num[None]]
            fluid_index = np.array(np.where(np_material == const.MATERIAL_FLUID))
            solid_index = np.array(np.where(np_material == const.MATERIAL_SOLID))
            if len(fluid_index[0]) > 0:
                writer = ti.PLYWriter(num_vertices=len(fluid_index[0]))
                writer.add_vertex_pos(pos[fluid_index][0][:, 0], pos[fluid_index][0][:, 1], pos[fluid_index][0][:, 2])
                writer.export_frame(frame, 'D:\\实验ply\\ply_202204048' + '\\fluid.ply')

            if len(solid_index[0]) > 0:
                writer = ti.PLYWriter(num_vertices=len(solid_index[0]))
                writer.add_vertex_pos(pos[solid_index][0][:, 0], pos[solid_index][0][:, 1], pos[solid_index][0][:, 2])
                writer.export_frame(frame, 'D:\\实验ply\\ply_202204048' + '\\solid.ply')
