import taichi as ti


@ti.data_oriented
class CollisionDetection:
    def __init__(self,
                 grid_num,
                 particle_type,
                 ):
        self.grid_num = grid_num
        self.dx = 1 / (self.grid_num - 1)
        self.inv_dx = 1 / self.dx
        self.particle_type = particle_type
        # 三角形顶点信息
        self.triangle_vertices = ti.Vector.field(3, ti.f32)
        # 三角形顶点索引信息
        self.triangle_indices = ti.field(ti.i32)
        # 三角形法线信息
        self.triangle_normals = ti.Vector.field(3, ti.f32)
        # 三角形顶点法线索引信息
        self.triangle_normals_indices = ti.field(ti.i32)
        # 固体符号距离场
        self.sign_distance_field = ti.field(ti.f32, shape=(self.grid_num,) * 3)
        self.neighbour = (3,) * 3

    # 将顶点信息和顶点坐标信息初始化
    def init_collide_triangle(self, vertices_num, indices_num, normals_num, _normals_indices_num):
        ti.root.dense(ti.i, vertices_num).place(self.triangle_vertices)
        ti.root.dense(ti.i, indices_num).place(self.triangle_indices)
        ti.root.dense(ti.i, normals_num).place(self.triangle_normals)
        ti.root.dense(ti.i, _normals_indices_num).place(self.triangle_normals_indices)

    def numpy_to_field(self, vertices, indices, normals, _normals_indices):
        self.triangle_vertices.from_numpy(vertices)
        self.triangle_indices.from_numpy(indices)
        self.triangle_normals_indices.from_numpy(_normals_indices)
        self.triangle_normals.from_numpy(normals)

    # 点到三角形的最短距离 https://zhuanlan.zhihu.com/p/148511581
    @ti.func
    def point_triangle_distance_min(self, A, B, C, P):
        # nearest_point = ti.Vector([0.0, 0.0, 0.0])
        # min_distance = 10.0
        e0 = B - A
        e1 = C - A
        dif = A - P
        a = e0.dot(e0)
        b = e0.dot(e1)
        c = e1.dot(e1)
        d = e0.dot(dif)
        e = e1.dot(dif)
        f = dif.dot(dif)
        det = a * c - b * b
        sc = (b * e - c * d)
        tc = (b * d - a * e)
        # if b1 and b2 and b3:
        #     min_distance = ti.sqrt(a * sc * sc + 2 * b * sc * tc + c * tc * tc + 2 * d * sc + 2 * e * tc + f)
        #     nearest_point = A + sc * e0 + tc * e1
        if sc + tc <= det:
            if sc < .0:
                if tc < .0:
                    # 4
                    if d < .0:
                        tc = .0
                        sc = 1.0 if -d >= a else -d / a
                    else:
                        sc = .0
                        temp = 1.0 if -e >= c else -e / c
                        tc = 0.0 if e >= 0.0 else temp
                else:
                    sc = 0
                    temp = 1.0 if -e >= c else -e / c
                    tc = 0.0 if e >= 0.0 else temp
            else:
                if tc < 0.0:
                    # 5
                    tc = 0.0
                    temp = 1.0 if -d >= a else -d / a
                    sc = 0.0 if d >= 0.0 else temp
                else:
                    # 0
                    inv_det = 1.0 / det
                    sc *= inv_det
                    tc *= inv_det
        else:
            if sc < 0.0:
                # 2
                tmp0 = b + d
                tmp1 = c + e
                if tmp1 > tmp0:
                    number = tmp1 - tmp0
                    denom = a - 2.0 * b + c
                    sc = 1.0 if number >= denom else number / denom
                    tc = 1.0 - sc
                else:
                    sc = 0.0
                    temp = 0.0 if e >= 0.0 else -e / c
                    tc = 1.0 if tmp1 <= 0.0 else temp
            elif tc < 0.0:
                # 6
                tmp0 = b + e
                tmp1 = a + d
                if tmp1 > tmp0:
                    numer = c + e - b - d
                    if numer <= 0.0:
                        sc = 0.0
                    else:
                        denom = a - 2.0 * b + c
                        sc = 1.0 if numer >= denom else numer / denom
                    tc = 1.0 - sc
                else:
                    tc = 0.0
                    temp = 1.0 if -d >= a else -d / a
                    sc = 0.0 if d >= 0.0 else temp
            else:
                numer = c + e - b - d
                if numer <= 0.0:
                    sc = 0.0
                else:
                    denom = a - 2.0 * b + c
                    sc = 1.0 if numer >= denom else numer / denom
                tc = 1.0 - sc
        min_distance = sc * (a * sc + 2.0 * b * tc + 2.0 * d) + tc * (c * tc + 2.0 * e) + f
        nearest_point = A + sc * e0 + tc * e1
        # min_distance = (P - nearest_point).norm()

        return min_distance, nearest_point

    @ti.kernel
    def init_level_set(self):
        # 遍历每个网格节点，找到每个节点距离三角形网格的最短距离
        for I in ti.grouped(self.sign_distance_field):
            # 每个节点的位置
            i, j, k = I
            # index = i + j * (self.grid_num + 1) + k * (self.grid_num + 1) ** 2

            min_dis = 10.0
            nearest_point = ti.Vector([0.0, 0.0, 0.0])
            min_index = 0
            # 网格节点位置
            node_pos = I * self.dx
            for triangle in range(self.triangle_indices.shape[0] / 3):
                A = self.triangle_vertices[self.triangle_indices[triangle * 3]]
                B = self.triangle_vertices[self.triangle_indices[triangle * 3 + 1]]
                C = self.triangle_vertices[self.triangle_indices[triangle * 3 + 2]]
                distance, min_point = self.point_triangle_distance_min(A, B, C, node_pos)
                if distance < min_dis:
                    min_dis = distance
                    nearest_point = min_point
                    min_index = triangle * 3
            # 判断正负
            n1 = self.triangle_normals[self.triangle_normals_indices[min_index]]
            n2 = self.triangle_normals[self.triangle_normals_indices[min_index + 1]]
            n3 = self.triangle_normals[self.triangle_normals_indices[min_index + 2]]
            n = (n1 + n2 + n3) / 3
            sign = n.dot(node_pos - nearest_point)
            if sign >= 0:
                self.sign_distance_field[I] = min_dis
            else:
                self.sign_distance_field[I] = -min_dis

    @ti.func
    def collision(self, position, velocity):
        # 查找到距离最近的三角形
        min_dis = 10.0
        nearest_point = ti.Vector([0.0, 0.0, 0.0])
        min_index = 0
        # 网格节点位置
        for triangle in range(self.triangle_indices.shape[0] / 3):
            A = self.triangle_vertices[self.triangle_indices[triangle * 3]]
            B = self.triangle_vertices[self.triangle_indices[triangle * 3 + 1]]
            C = self.triangle_vertices[self.triangle_indices[triangle * 3 + 2]]
            distance, min_point = self.point_triangle_distance_min(A, B, C, position)
            if distance < min_dis:
                # 最短距离
                min_dis = distance
                # 最近点
                nearest_point = min_point
                # 最近三角形的下标
                min_index = triangle * 3
        # 最近三角形的法线
        n1 = self.triangle_normals[self.triangle_normals_indices[min_index]]
        n2 = self.triangle_normals[self.triangle_normals_indices[min_index + 1]]
        n3 = self.triangle_normals[self.triangle_normals_indices[min_index + 2]]
        normal = (n1 + n2 + n3) / 3.0
        # d = velocity.dot(normal)
        # new_vel = velocity - (d * normal)
        new_pos = nearest_point
        new_vel = (velocity - (ti.dot(velocity, normal) * normal))
        if min_dis == 0.0:
            new_pos = position

        return new_pos, new_vel

    # 三次线性插值函数
    @ti.func
    def linear_interpolation_sdf(self, pos: ti.template()):
        base = (pos * self.inv_dx).cast(int)
        fx = pos * self.inv_dx - base.cast(float)
        w = [(1 - fx) * self.dx, fx * self.dx]
        result = 0.0
        for i, j, k in ti.static(ti.ndrange(2, 2, 2)):
            weight = w[i][0] * w[j][1] * w[k][2] * self.inv_dx * self.inv_dx * self.inv_dx
            offset = [i, j, k]
            result += self.sign_distance_field[base + offset] * weight
        return result
