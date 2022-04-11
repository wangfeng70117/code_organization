[TOC]



# 构建表面

## marching_squares.py

构建2D的MarchingCube显示曲面。具体MC原理在我的知乎[流体仿真构建MarchingCube、LevelSet以及两者相互转化 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/495674830)

### 方法及参数：

#### class参数：

grid_num:  空间网格分辨率。

#### update_field(field: ti.template())：更新标量场

参数field: 更新构建MarchingCube的标量场

#### create_mc: 生成MarchingCube

创建MarchingCube

## level_set.py

### 方法及参数：

#### class参数

grid_num: 空间网格分辨率。

#### gen_level_set(particle_position: ti.template(), create_particle_num: int): 构建SDF

 参数particle_position: 所有粒子的位置；create_particle_num：已创建粒子数量。

这个方法及其耗费性能，因为需要遍历每个网格节点到每个粒子的最短距离，时间复杂度是O(n^2)。我并没有研究如何去高效的计算SDF。

这里说明一下为什么这么写。

因为taichi的field在定义时必须指定shape，但是有的时候我们可能一边运行一边添加新的粒子，这种情况就不能在初始时定义field长度。所以定义了一个最大的field长度，然后每创建一个粒子更新create_particle_num，然后根据create_particle_num来遍历粒子进行仿真。

#### calculate_gradient(): 计算SDF的梯度

这里也可以计算其他标量场的梯度，只需要把其中的sign_distance_field改成你的标量场就可以了。

SDF的梯度就是流体表面的法向量。

#### calculate_gradient(): 计算SDF的拉普拉斯算子

这里也可以计算其他标量场的拉普拉斯算子，只需要把其中的sign_distance_field改成你的标量场就可以了。

SDF的拉普拉斯算子就是流体表面的曲率。

计算散度和旋度我没有写，但是大致是差不多的，求解方法在《Fluid Simulation for computer graphics》这本书的后边有写。

#### bilinear_gradient(pos)：使用双线性插值函数求解空间内任意一点的梯度值。

双线性差值原理在GAMES201里有讲。

#### bilinear_laplacian(pos)：使用双线性插值函数求解空间内任意一点的拉普拉斯算子。

## fluid_surface.py

3D的流体表面构建，用法和2D基本一样，下边只列出2D没有的方法。

#### discrete_triangles()：将MarchingCube构建的三角形离散为点坐标。

#### discrete_triangle(A, B, C)：将三角形离散为点坐标。

参数为三角形的三个顶点坐标，其中discrete_num为离散点的密度。

# 读取obj以及模型操作

## read_obj.py

读取obj模型。

obj文件格式：

v：顶点位置

vn：顶点法线方向

f：三角面信息

f: a1//b1//c1 a2//b2//c2 a3//b3//c3 a:顶点索引 b纹理索引 c法线索引，三个法线索引的平均值就是面的法线方向.

即：a1a2a3为三角形的三个顶点索引值，c1c2c3为三角形的三个顶点法向量索引值

### 方法及参数：

#### read_obj(file_name): file_name: obj文件名。

这里我将模型的顶点坐标映射到了0-1之间，因为mpm代码是将世界坐标固定到了0-1，方便将粒子导入到仿真算法。

返回值：

vertices：顶点坐标二维数组。

indexes：顶点坐标索引一维数组；每三个值表示一个三角形三个顶点的索引。

normals：顶点法向量二维数组。

normal_indices：顶点法向量索引一维数组，每三个值表示一个三角形三个顶点法向量的索引。

## find_nearest_point_and_distance.py

找到空间内某一点到三角形的最短距离以及最近点位置。

这个文件可以直接运行，可以按tfgh四个按键移动粒子进行观察。

原理：https://zhuanlan.zhihu.com/p/148511581

### 方法及参数：

#### point_triangle_distance_min(A, B, C, P):点P到三角形ABC的最短距离

返回最短距离以及最近点位置。

## 

将abq文件内的点坐标转化为numpy数组，以便于导入mpm求解器。

因为mpm算法需要模型的粒子位置，我自己写的obj模型离散化代码效率太低了，离散一个模型可能需要几个小时才能跑完，所以就用了Large Modal Deformation Factory这个软件来离散化obj模型，但是这个模型导出的是一种abq文件，这个文件内记录的都是点坐标，所以我就写了一份将abq模型转化为numpy数组的代码。

代码修改file_name然后直接运行就行。

同样，我也将坐标转化到了0-1，方便导入到mpm求解器。

# mpm_examples

一个3Dmpm程序，将obj模型转化为mpm粒子并进行模拟。

## mpm_solver.py

这个代码使用Large Modal Deformation Factory导出的abq文件，使用read_abq.py转化为.npy文件，然后转化为mpm粒子进行仿真。

和上边的原因一样，因为taichi的field在定义时必须指定shape，但是有的时候我们可能一边运行一边添加新的粒子，这种情况就不能在初始时定义field长度。所以就指定了一个shape为len(.npy)长度的field，然后动态赋值给mpm里的particles。

### 方法及参数

#### generate_bear()：将.npy转化为field后的粒子位置，赋值给流体粒子。

在这个代码初始化的时候，创建了shape为len(.npy)长度的field，这里就是将这个field赋值给particles，从而进行仿真。

#### init_bear(): 将通过abp文件生成的npy文件转化为taichi的field

这里的from_numpy方法，一定要在kernel运行之前执行，因为taichi在进行field计算之后不允许再声明field。

#### run中将粒子不同的粒子分别导出到ply文件

因为在使用houdini进行渲染时，同一种ply文件只能赋予相同材质，如果将流体和固体导入到同一个ply中，就不能分别渲染。

# ReadObj

将obj模型作为三角面片导入到mpm系统，进行最简单的碰撞检测。

## collision_detection.py

碰撞检测算法。我将模拟空间分割为grid_num ** 3 的网格，然后通过插值函数，判断粒子新的位置的SDF是否为负数，如果当前粒子所在网格的SDF为负数，表示粒子发生了碰撞，进行碰撞检测算法。

#### init_collide_triangle(vertices_num, indices_num, normals_num, _normals_indices_num)

初始化field的shape，将obj模型转化出来的顶点等信息，转化为taichi的field。

因为不同模型的的经典信息不同，所以这里不在初始化的时候写field的shape，而是先解析obj文件，再根据解析出来的数组长度赋值。具体的文档在这里。

[Fields (advanced) | Taichi Docs](https://docs.taichi.graphics/lang/articles/layout)

这个方法一定要在程序操作field之前进行，因为如果taichi对field操作之后，无法再初始化或者创建新的field，也不能再修改field的shape.

#### numpy_to_field(self, vertices, indices, normals, _normals_indices): 将.npy文件转化为field

同样，必须在程序操作field之前进行。

#### init_level_set()：初始化obj模型的level_set

效率也很低，也是每个网格节点遍历每个粒子，好在只需要在程序初始化的时候执行一次，可以接受。

由于obj模型内包含顶点法向量，所以当n.dot(node_pos - nearest_point)为负的时候，表明当前网格节点在模型内部，为正的时候表示当前网格节点在模型外部。从而判断SDF的正负。

#### collision(position, velocity):碰撞检测算法

如果发生碰撞，检测距离当前粒子最近的三角面，然后通过碰撞检测算法修改粒子的速度和位置。

碰撞后的速度：new_vel = (velocity - (ti.dot(velocity, normal) * normal))

我这里没有加速度衰减，如果需要衰减，成一个衰减系数就可以。

而且我直接把法线方向的速度弄没了，如果需要，可以参考《Fluid Engine Development》这本书里有将，网上也有很多资料，很容易理解。

#### linear_interpolation_sdf(pos): 线性插值函数
