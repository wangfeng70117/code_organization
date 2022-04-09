import taichi as ti


@ti.data_oriented
class Field:
    def __init__(self, diff_n_grid):
        self.diff_n_grid = diff_n_grid
        self.diff_dx = 1 / self.diff_n_grid
        self.diff_inv_dx = 1 / self.diff_dx
        self.m_field = ti.field(ti.f32, shape=(self.diff_n_grid,) * 3)
        self.gradient = ti.Vector.field(3, ti.f32, shape=(self.diff_n_grid,) * 3)
        self.divergence = ti.field(ti.f32, shape=(self.diff_n_grid,) * 3)
        self.laplacian = ti.field(ti.f32, shape=(self.diff_n_grid,) * 3)

    @ti.kernel
    def init(self):
        for I in ti.grouped(self.m_field):
            self.m_field[I] = .0

    @ti.kernel
    def update(self, field: ti.template()):
        for I in ti.grouped(self.m_field):
            self.m_field[I] = field[I]

    @ti.func
    def ti_vector(self, x, y):
        return ti.Vector([x, y])

    # 中心差分计算梯度场
    @ti.kernel
    def calculate_gradient(self):
        for I in ti.grouped(self.m_field):
            i, j, k = I
            u, v, w = .0, .0, .0
            # 判断边界条件
            if i == 0:
                u = (self.m_field[i + 1, j, k] - self.m_field[i, j, k]) * 0.5 * self.diff_dx
            elif i == self.diff_n_grid - 1:
                u = (self.m_field[i, j, k] - self.m_field[i - 1, j, k]) * 0.5 * self.diff_dx
            else:
                u = (self.m_field[i + 1, j, k] - self.m_field[i - 1, j, k]) * 0.5 * self.diff_dx

            if j == 0:
                v = (self.m_field[i, j + 1, k] - self.m_field[i, j, k]) * 0.5 * self.diff_dx
            elif j == self.diff_n_grid - 1:
                v = (self.m_field[i, j, k] - self.m_field[i, j - 1, k]) * 0.5 * self.diff_dx
            else:
                v = (self.m_field[i, j + 1, k] - self.m_field[i, j - 1, k]) * 0.5 * self.diff_dx

            if k == 0:
                w = (self.m_field[i, j, k + 1] - self.m_field[i, j, k]) * 0.5 * self.diff_dx
            elif k == self.diff_n_grid - 1:
                w = (self.m_field[i, j, k] - self.m_field[i, j, k - 1]) * 0.5 * self.diff_dx
            else:
                w = (self.m_field[i, j, k + 1] - self.m_field[i, j, k - 1]) * 0.5 * self.diff_dx
            self.gradient[I] = ti.Vector([u, v, w]).normalized()

    # 中心差分计算拉普拉斯算子
    @ti.kernel
    def calculate_laplacian(self):
        for I in ti.grouped(self.m_field):
            i, j, k = I
            u, v, w = .0, .0, .0
            if i == 0:
                u = (self.m_field[i + 1, j, k] - self.m_field[
                    i, j, k]) * self.diff_dx * self.diff_dx
            elif i == self.diff_n_grid - 1:
                u = (-self.m_field[i, j, k] + self.m_field[
                    i - 1, j, k]) * self.diff_dx * self.diff_dx
            else:
                u = (self.m_field[i + 1, j, k] - 2 * self.m_field[i, j, k] +
                     self.m_field[i - 1, j, k]) * self.diff_dx * self.diff_dx

            if j == 0:
                v = (self.m_field[i, j + 1, k] - self.m_field[
                    i, j, k]) * self.diff_dx * self.diff_dx
            elif j == self.diff_n_grid - 1:
                v = (-self.m_field[i, j, k] + self.m_field[
                    i, j - 1, k]) * self.diff_dx * self.diff_dx
            else:
                v = (self.m_field[i, j + 1, k] - 2 * self.m_field[i, j, k] +
                     self.m_field[i, j - 1, k]) * self.diff_dx * self.diff_dx

            if k == 0:
                w = (self.m_field[i, j, k + 1] - self.m_field[
                    i, j, k]) * self.diff_dx * self.diff_dx
            elif k == self.diff_n_grid - 1:
                w = (-self.m_field[i, j, k] + self.m_field[
                    i, j, k - 1]) * self.diff_dx * self.diff_dx
            else:
                w = (self.m_field[i, j, k + 1] - 2 * self.m_field[i, j, k] +
                     self.m_field[i, j, k - 1]) * self.diff_dx * self.diff_dx
            self.laplacian[I] = u + v + w

    # 差值函数
    @ti.kernel
    def get_position_laplacian(self, pos: ti.template()):
        base = (pos * self.diff_inv_dx - 0.5).cast(int)
        fx = pos * self.diff_inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        value = .0
        for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
            weight = 1.0
            for i in ti.static(range(3)):
                weight *= w[offset[i]][i]
            value += self.laplacian[base + offset] * weight
        return value

    # 差值函数
    @ti.kernel
    def get_position_gradient(self, pos: ti.template()):
        base = (pos * self.diff_inv_dx - 0.5).cast(int)
        fx = pos * self.diff_inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        value = ti.Vector([.0, .0])
        for offset in ti.static(ti.grouped(ti.ndrange(*self.neighbour))):
            weight = 1.0
            for i in ti.static(range(3)):
                weight *= w[offset[i]][i]
            value += self.gradient[base + offset] * weight
        return value
