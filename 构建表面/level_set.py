import taichi as ti


@ti.data_oriented
class LevelSet:
    def __init__(self, grid_num):
        self.grid_num = grid_num
        self.diff_dx = 1 / self.grid_num
        self.diff_inv_dx = 1 / self.diff_dx
        self.radius = self.diff_dx
        self.sign_distance_field = ti.field(ti.f32, shape=(self.grid_num, self.grid_num))
        self.gradient = ti.Vector.field(2, ti.f32, shape=(self.grid_num, self.grid_num))
        self.divergence = ti.field(ti.f32, shape=(self.grid_num, self.grid_num))
        self.laplacian = ti.field(ti.f32, shape=(self.grid_num, self.grid_num))

    # 生成level set隐式曲面
    @ti.kernel
    def gen_level_set(self, particle_position: ti.template(),
                      create_particle_num: int):
        for i, j in ti.ndrange(self.grid_num, self.grid_num):
            min_dis = 10.0
            node_pos = ti.Vector([i * self.diff_dx, j * self.diff_dx])
            for I in range(create_particle_num):
                distance = (particle_position[I] - node_pos).norm() - self.radius
                if distance < min_dis:
                    min_dis = distance
            self.sign_distance_field[i, j] = min_dis

    @ti.func
    def ti_vector(self, x, y):
        return ti.Vector([x, y])

    # 中心差分计算梯度场
    @ti.kernel
    def calculate_gradient(self):
        for I in ti.grouped(self.sign_distance_field):
            i, j = I
            u, v = .0, .0
            # 区分边界位置
            if i == 0:
                u = (self.sign_distance_field[i + 1, j] - self.sign_distance_field[i, j]) * 0.5 * self.diff_inv_dx
            elif i == self.grid_num - 1:
                u = (self.sign_distance_field[i, j] - self.sign_distance_field[i - 1, j]) * 0.5 * self.diff_inv_dx
            else:
                u = (self.sign_distance_field[i + 1, j] - self.sign_distance_field[i - 1, j]) * 0.5 * self.diff_inv_dx

            if j == 0:
                v = (self.sign_distance_field[i, j + 1] - self.sign_distance_field[i, j]) * 0.5 * self.diff_inv_dx
            elif j == self.grid_num - 1:
                v = (self.sign_distance_field[i, j] - self.sign_distance_field[i, j - 1]) * 0.5 * self.diff_inv_dx
            else:
                v = (self.sign_distance_field[i, j + 1] - self.sign_distance_field[i, j - 1]) * 0.5 * self.diff_inv_dx

            self.gradient[I] = self.ti_vector(u, v).normalized()

    # 中心差分计算拉普拉斯算子
    @ti.kernel
    def calculate_laplacian(self):
        for I in ti.grouped(self.sign_distance_field):
            i, j = I
            u, v = .0, .0
            if i == 0:
                u = (self.sign_distance_field[i + 1, j] - 2 * self.sign_distance_field[i, j] + self.sign_distance_field[
                    i, j]) * self.diff_inv_dx * self.diff_inv_dx
            elif i == self.grid_num - 1:
                u = (self.sign_distance_field[i, j] - 2 * self.sign_distance_field[i, j] + self.sign_distance_field[
                    i - 1, j]) * self.diff_inv_dx * self.diff_inv_dx
            else:
                u = (self.sign_distance_field[i + 1, j] - 2 * self.sign_distance_field[i, j] + self.sign_distance_field[
                    i - 1, j]) * self.diff_inv_dx * self.diff_inv_dx

            if j == 0:
                v = (self.sign_distance_field[i, j + 1] - 2 * self.sign_distance_field[i, j] + self.sign_distance_field[
                    i, j]) * self.diff_inv_dx * self.diff_inv_dx
            elif j == self.grid_num - 1:
                v = (self.sign_distance_field[i, j] - 2 * self.sign_distance_field[i, j] + self.sign_distance_field[
                    i, j - 1]) * self.diff_inv_dx * self.diff_inv_dx
            else:
                v = (self.sign_distance_field[i, j + 1] - 2 * self.sign_distance_field[i, j] + self.sign_distance_field[
                    i, j - 1]) * self.diff_inv_dx * self.diff_inv_dx

            self.laplacian[I] = u + v

    # 双线性差值函数
    @ti.func
    def bilinear_laplacian(self, pos: ti.template()):
        base = (pos * self.diff_inv_dx).cast(int)
        fx = pos * self.diff_inv_dx - base.cast(float)
        w = [(1 - fx) * self.diff_dx, fx * self.diff_dx]
        value = .0
        for i, j in ti.static(ti.ndrange(2, 2)):
            offset = ti.Vector([i, j])
            weight = w[i][0] * w[j][1] * self.diff_inv_dx * self.diff_inv_dx
            value += self.laplacian[base + offset] * weight
        return value

    # 双线性差值函数
    @ti.func
    def bilinear_gradient(self, pos: ti.template()):
        base = (pos * self.diff_inv_dx).cast(int)
        fx = pos * self.diff_inv_dx - base.cast(float)
        w = [(1 - fx) * self.diff_dx, fx * self.diff_dx]
        value = ti.Vector([.0, .0])
        for i, j in ti.static(ti.ndrange(2, 2)):
            offset = ti.Vector([i, j])
            weight = w[i][0] * w[j][1] * self.diff_inv_dx * self.diff_inv_dx
            value += self.gradient[base + offset] * weight
        return value
