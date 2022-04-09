import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

grid_num = 20
dx = 1 / grid_num


def point_triangle_distance_min(A, B, C, P):
    # nearest_point = np.Vector([0.0, 0.0, 0.0])
    # min_distance = 10.0
    e0 = B - A
    e1 = C - A
    dif = A - P
    a = np.dot(e0, e0)
    b = np.dot(e0, e1)
    c = np.dot(e1, e1)
    d = np.dot(e0, dif)
    e = np.dot(e1, dif)
    f = np.dot(dif, dif)
    det = a * c - b * b
    sc = (b * e - c * d)
    tc = (b * d - a * e)
    # if b1 and b2 and b3:
    #     min_distance = np.sqrt(a * sc * sc + 2 * b * sc * tc + c * tc * tc + 2 * d * sc + 2 * e * tc + f)
    #     nearest_point = A + sc * e0 + tc * e1
    if sc + tc <= det:
        if sc < 0:
            if tc < 0:
                # 4
                if d < 0:
                    tc = 0
                    sc = 1 if -d >= a else -d / a
                else:
                    sc = 0
                    temp = 1 if -e >= c else -e / c
                    tc = 0 if e >= 0 else temp
            else:
                sc = 0
                temp = 1 if -e >= c else -e / c
                tc = 0 if e >= 0 else temp
        else:
            if tc < 0:
                # 5
                tc = 0
                temp = 1 if -d >= a else -d / a
                sc = 0 if d >= 0 else temp
            else:
                # 0
                inv_det = 1 / det
                sc *= inv_det
                tc *= inv_det
    else:
        if sc < 0:
            # 2
            tmp0 = b + d
            tmp1 = c + e
            if tmp1 > tmp0:
                number = tmp1 - tmp0
                denom = a - 2 * b + c
                sc = 1 if number >= denom else number / denom
                tc = 1 - sc
            else:
                sc = 0
                temp = 0 if e >= 0 else -e / c
                tc = 1 if tmp1 <= 0 else temp
        elif tc < 0:
            # 6
            tmp0 = b + e
            tmp1 = a + d
            if tmp1 > tmp0:
                numer = c + e - b - d
                if numer <= 0:
                    sc = 0
                else:
                    denom = a - 2 * b + c
                    sc = 1 if numer >= denom else numer / denom
                tc = 1 - sc
            else:
                tc = 0
                temp = 1 if -d >= a else -d / a
                sc = 0 if d >= 0 else temp
        else:
            numer = c + e - b - d
            if numer <= 0:
                sc = 0
            else:
                denom = a - 2 * b + c
                sc = 1 if numer >= denom else numer / denom
            tc = 1 - sc
    min_distance = sc * (a * sc + 2 * b * tc + 2 * d) + tc * (c * tc + 2 * e) + f
    nearest_point = A + sc * e0 + tc * e1
    return min_distance, nearest_point


# obj文件格式：
# v：顶点位置
# vn：顶点法线方向
# f: a//b//c a//b//c a//b//c a:顶点索引 b纹理索引 c法线索引，三个法线索引的平均值就是面的法线方向.
# 返回值： 顶点坐标、 顶点索引、 法线坐标、 法线索引
def read_obj(file_name):
    with open(file_name) as file:
        vertex = []
        index = []
        normal_index = []
        normal = []
        while True:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                vertex.append((float(strs[2]), float(strs[3]), float(strs[4])))
            if strs[0] == "vn":
                normal.append((float(strs[1]), float(strs[2]), float(strs[3])))
            if strs[0] == "f":
                s1 = strs[1].split("/")[0]
                s2 = strs[2].split("/")[0]
                s3 = strs[3].split("/")[0]
                index.append(int(s1) - 1)
                index.append(int(s2) - 1)
                index.append(int(s3) - 1)
                # index.append((int(s1), int(s2), int(s3)))
                n1 = strs[1].split("/")[2]
                n2 = strs[2].split("/")[2]
                n3 = strs[3].split("/")[2]
                normal_index.append(int(n1) - 1)
                normal_index.append(int(n2) - 1)
                normal_index.append(int(n3) - 1)

    vertices = np.array(vertex)
    indexes = np.array(index)
    normal_indices = np.array(normal_index)
    normals = np.array(normal)
    # 将顶点坐标映射到0-1
    x_list = []
    y_list = []
    z_list = []
    for i in range(len(vertices)):
        x_list.append(vertices[i][0])
        y_list.append(vertices[i][1])
        z_list.append(vertices[i][2])
    # 找到每个坐标轴映射需要的比例
    max_x = max(x_list)
    min_x = min(x_list)
    max_y = max(y_list)
    min_y = min(y_list)
    max_z = max(z_list)
    min_z = min(z_list)
    xc = max_x - min_x
    yc = max_y - min_y
    zc = max_z - min_z
    rate = max(xc, yc, zc)

    x_re = 0
    y_re = 0
    z_re = 0
    if min_x < 0:
        x_re = - min_x
    if min_y < 0:
        y_re = - min_y
    if min_z < 0:
        z_re = - min_z
    for i in range(len(vertices)):
        vertices[i][0] += x_re
        vertices[i][1] += y_re
        vertices[i][2] += z_re
    for i in range(len(vertices)):
        vertices[i] /= rate

    return vertices, indexes, normals, normal_indices

# read_obj("cirque.obj")
