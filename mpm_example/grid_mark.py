import taichi as ti

from mpm_example.const import const


@ti.data_oriented
class GridMark:
    def __init__(self,
                 grid_num
                 ):
        self.grid_num = grid_num
        self.dx = 1 / self.grid_num
        self.inv_dx = self.grid_num
        self.node = ti.Struct.field({
            "grid_pos": ti.types.vector(3, ti.f32),
            "grid_color": ti.types.vector(3, ti.f32),
            "mark": ti.i32,
            "fluid_bound": ti.i32,
            "solid_bound": ti.i32,
            "contact_bound": ti.i32,
        }, shape=(self.grid_num,) * 3)
        self.neighbour = (3,) * 3

    @ti.kernel
    def init(self):
        for I in ti.grouped(self.m_field):
            self.m_field[I] = .0

    @ti.func
    def get_index(self, base):
        i, j, k = base
        return i + j * (self.grid_num + 1) + k * (self.grid_num + 1) ** 2

    @ti.kernel
    def init_mark(self):
        for I in ti.grouped(self.node):
            self.node[I].mark = 0
            self.node[I].fluid_bound = 0
            self.node[I].solid_bound = 0
            self.node[I].contact_bound = 0

    @ti.kernel
    def update_mark(self, particles: ti.template(), particle_num: int):
        # 每帧将标记为初始化为空气
        for I in ti.grouped(self.node):
            self.node[I].mark = const.MARK_AIR
        # 找到流体粒子所在网格，标记流体区域
        for p in range(particle_num):
            if particles[p].material == const.MATERIAL_FLUID:
                base = (particles[p].position * self.inv_dx).cast(int)
                self.node[base].mark = const.MARK_FLUID
        # 标记固体区域
        for p in range(particle_num):
            if particles[p].material == const.MATERIAL_SOLID:
                base = (particles[p].position * self.inv_dx).cast(int)
                self.node[base].mark = const.MARK_SOLID

    @ti.kernel
    def calculate_bound(self, particles: ti.template(), particle_num: int):
        # 判断边界粒子
        for p in range(particle_num):
            # 流体粒子的边界
            if particles[p].material == const.MATERIAL_FLUID:
                base = (particles[p].position * self.inv_dx - 0.5).cast(int)
                for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                    if self.node[base + offset].mark != const.MARK_FLUID:
                        self.node[base].fluid_bound = 1
            if particles[p].material == const.MATERIAL_SOLID:
                base = (particles[p].position * self.inv_dx).cast(int)
                for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
                    if self.node[base + offset].mark != const.MARK_SOLID:
                        self.node[base].solid_bound = 1
        # 判定接触边界
        for I in ti.grouped(self.node):
            if self.node[I].fluid_bound == self.node[I].solid_bound == 1:
                self.node[I].contact_bound = 1
