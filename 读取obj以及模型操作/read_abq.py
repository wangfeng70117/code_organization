import numpy as np
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
file_name = "bear_less"
points = []
x_list = []
y_list = []
z_list = []
for line in open(file_name + ".abq", "r"):
    if line.startswith('*NODE'):
        print(line)
        continue

    if line.startswith('*ELEMENT'):
        print(line)
        break
    strs = line.split(", ")
    x_list.append(float(strs[1]))
    y_list.append(float(strs[2]))
    z_list.append(float(strs[3]))

min_x = min(x_list)
min_y = min(y_list)
min_z = min(z_list)

max_x = max(x_list)
max_y = max(y_list)
max_z = max(z_list)

xc = max_x - min_x
yc = max_y - min_y
zc = max_z - min_z
rate = max(xc, yc, zc)


for i in range(len(x_list)):
    x_list[i] += - min_x
    y_list[i] += - min_y
    z_list[i] += - min_z

    x_list[i] = x_list[i] / rate
    y_list[i] = y_list[i] / rate
    z_list[i] = z_list[i] / rate
for i in range(len(x_list)):
    points.append([x_list[i], y_list[i], z_list[i]])

np.save(file_name + ".npy", points)

# fig = plt.figure()
# ax = Axes3D(fig)
# ax.scatter(x_list, y_list, z_list)
#
# # 添加坐标轴(顺序是Z, Y, X)
# ax.set_zlabel('Z', fontdict={'size': 15, 'color': 'red'})
# ax.set_ylabel('Y', fontdict={'size': 15, 'color': 'red'})
# ax.set_xlabel('X', fontdict={'size': 15, 'color': 'red'})
# plt.show()