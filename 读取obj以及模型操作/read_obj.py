import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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


read_obj("cirque.obj")