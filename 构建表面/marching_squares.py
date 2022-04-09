import taichi as ti
import numpy as np


@ti.data_oriented
class MarchingSquares:
    def __init__(self, grid_num):
        self._et = np.array(
            [
                [[-1, -1], [-1, -1]],  #
                [[0, 1], [-1, -1]],  # a
                [[0, 2], [-1, -1]],  # b
                [[1, 2], [-1, -1]],  # ab
                [[1, 3], [-1, -1]],  # c
                [[0, 3], [-1, -1]],  # ca
                [[1, 3], [0, 2]],  # cb
                [[2, 3], [-1, -1]],  # cab
                [[2, 3], [-1, -1]],  # d
                [[2, 3], [0, 1]],  # da
                [[0, 3], [-1, -1]],  # db
                [[1, 3], [-1, -1]],  # dab
                [[1, 2], [-1, -1]],  # dc
                [[0, 2], [-1, -1]],  # dca
                [[0, 1], [-1, -1]],  # dcb
                [[-1, -1], [-1, -1]],  # dcab
            ],
            np.int32)
        self.et = ti.Vector.field(2, int, self._et.shape[:2])
        self.et.from_numpy(self._et)

        self.grid_num = grid_num
        self.dx = 1 / self.grid_num
        self.inv_dx = 1 / self.dx
        # 生成的边
        self.edge = ti.Struct.field({
            "begin_point": ti.types.vector(2, ti.f32),
            "end_point": ti.types.vector(2, ti.f32)
        }, shape=grid_num ** 2)
        self.edge_num = ti.field(int, shape=())
        self.m_field = ti.field(ti.f32, shape=(grid_num, grid_num))

    # 将隐式曲面通过marching cube转化为显示曲面

    @ti.kernel
    def update_field(self, field: ti.template()):
        for I in ti.grouped(self.m_field):
            self.m_field[I] = field

    @ti.func
    def gen_edge_pos(self, i, j, e):
        a = self.m_field[i, j]
        b = self.m_field[i + 1, j]
        c = self.m_field[i, j + 1]
        d = self.m_field[i + 1, j + 1]
        base_grid_pos = self.dx * ti.Vector([i, j])
        result_pos = ti.Vector([.0, .0])
        if e == 0:
            result_pos = base_grid_pos + ti.Vector([(abs(a) / (abs(a) + abs(b))) * self.dx, 0])
        if e == 1:
            result_pos = base_grid_pos + ti.Vector([0, (abs(a) / (abs(a) + abs(c))) * self.dx])
        if e == 2:
            result_pos = base_grid_pos + ti.Vector([self.dx, (abs(b) / (abs(b) + abs(d))) * self.dx])
        if e == 3:
            result_pos = base_grid_pos + ti.Vector([(abs(c) / (abs(c) + abs(d))) * self.dx, self.dx])
        return result_pos

    @ti.kernel
    def create_mc(self):
        self.edge_num[None] = 0
        for i, j in ti.ndrange(self.grid_num - 1, self.grid_num - 1):
            id = 0
            if self.m_field[i, j] > 0: id |= 1
            if self.m_field[i + 1, j] > 0: id |= 2
            if self.m_field[i, j + 1] > 0: id |= 4
            if self.m_field[i + 1, j + 1] > 0: id |= 8
            for k in ti.static(range(2)):
                if self.et[id, k][0] != -1:
                    n = ti.atomic_add(self.edge_num[None], 1)
                    self.edge[n].begin_point = self.gen_edge_pos(i, j, self.et[id, k][0])
                    self.edge[n].end_point = self.gen_edge_pos(i, j, self.et[id, k][1])
