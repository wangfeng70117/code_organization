#构建表面

##marching_squares.py

构建2D的MarchingCube显示曲面。具体MC原理在我的知乎[流体仿真构建MarchingCube、LevelSet以及两者相互转化 - 知乎 (zhihu.com)](https://zhuanlan.zhihu.com/p/495674830)

###方法及参数：

####class参数：

grid_num:  空间网格分辨率。

####update_field(field: ti.template())：更新标量场

参数field: 更新构建MarchingCube的标量场

####create_mc: 生成MarchingCube

创建MarchingCube

##level_set.py

###方法及参数：

####class参数

grid_num: 空间网格分辨率。

####gen_level_set(particle_position: ti.template(), create_particle_num: int): 构建SDF

 参数particle_position: 所有粒子的位置；create_particle_num：已创建粒子数量。

这个方法及其耗费性能，因为需要遍历每个网格节点到每个粒子的最短距离，时间复杂度是O(n^2)。我并没有研究如何去高效的计算SDF。

这里说明一下为什么这么写。

因为taichi的field在定义时必须指定shape，但是有的时候我们可能一边运行一边添加新的粒子，这种情况就不能在初始时定义field长度。所以定义了一个最大的field长度，然后每创建一个粒子更新create_particle_num，然后根据create_particle_num来遍历粒子进行仿真。

####calculate_gradient(): 计算SDF的梯度

这里也可以计算其他标量场的梯度，只需要把其中的sign_distance_field改成你的标量场就可以了。

SDF的梯度就是流体表面的法向量。

####calculate_gradient(): 计算SDF的拉普拉斯算子

这里也可以计算其他标量场的拉普拉斯算子，只需要把其中的sign_distance_field改成你的标量场就可以了。

SDF的拉普拉斯算子就是流体表面的曲率。

计算散度和旋度我没有写，但是大致是差不多的，求解方法在《Fluid Simulation for computer graphics》这本书的后边有写。

####bilinear_gradient(pos)：使用双线性插值函数求解空间内任意一点的梯度值。

双线性差值原理在GAMES201里有讲。

####bilinear_laplacian(pos)：使用双线性插值函数求解空间内任意一点的拉普拉斯算子。

##fluid_surface.py

3D的流体表面构建，用法和2D基本一样，下边只列出2D没有的方法。

####discrete_triangles()：将MarchingCube构建的三角形离散为点坐标。

####discrete_triangle(A, B, C)：将三角形离散为点坐标。

参数为三角形的三个顶点坐标，其中discrete_num为离散点的密度。

#读取obj以及模型操作

##read_obj.py

读取obj模型。

obj文件格式：

v：顶点位置

vn：顶点法线方向

f：三角面信息

f: a1//b1//c1 a2//b2//c2 a3//b3//c3 a:顶点索引 b纹理索引 c法线索引，三个法线索引的平均值就是面的法线方向.

即：a1a2a3为三角形的三个顶点索引值，c1c2c3为三角形的三个顶点法向量索引值

###方法及参数：

####read_obj(file_name): file_name: obj文件名。

这里我将模型的顶点坐标映射到了0-1之间，因为mpm代码是将世界坐标固定到了0-1，方便将粒子导入到仿真算法。

返回值：

vertices：顶点坐标二维数组。

indexes：顶点坐标索引一维数组；每三个值表示一个三角形三个顶点的索引。

normals：顶点法向量二维数组。

normal_indices：顶点法向量索引一维数组，每三个值表示一个三角形三个顶点法向量的索引。

##find_nearest_point_and_distance.py

找到空间内某一点到三角形的最短距离以及最近点位置。

这个文件可以直接运行，可以按tfgh四个按键移动粒子进行观察。

原理：https://zhuanlan.zhihu.com/p/148511581

###方法及参数：

####point_triangle_distance_min(A, B, C, P):点P到三角形ABC的最短距离

返回最短距离以及最近点位置。

##read_abq.py

将abq文件内的点坐标转化为numpy数组，以便于导入mpm求解器。

因为mpm算法需要模型的粒子位置，我自己写的obj模型离散化代码效率太低了，离散一个模型可能需要几个小时才能跑完，所以就用了Large Modal Deformation Factory这个软件来离散化obj模型，但是这个模型导出的是一种abq文件，这个文件内记录的都是点坐标，所以我就写了一份将abq模型转化为numpy数组的代码。

代码修改file_name然后直接运行就行。

同样，我也将坐标转化到了0-1，方便导入到mpm求解器。

#mpm_examples

一个3Dmpm程序，将obj模型转化为mpm粒子并进行模拟。

##mpm_solver.py

###方法及参数

####init_bear(): 将通过abp文件生成的npy文件转化为taichi的field

这里的from_numpy方法，一定要在kernel运行之前执行，因为taichi在进行field计算之后不允许再声明field。

####run中将粒子不同的粒子分别导出到ply文件

因为在使用houdini进行渲染时，同一种ply文件只能赋予相同材质，如果将流体和固体导入到同一个ply中，就不能分别渲染。