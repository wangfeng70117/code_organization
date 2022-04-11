import taichi as ti

from ReadObj.mpm_solver import MPMSolver
import ReadObj.read_obj as read_obj

# 调试的时候使用debug=True，可以检测数组下标越界，在调试时候用
# 调试数组下标越界、断言等
# 3D情况调试一个场，最好用ggui绘制出来
# 注意并行，如果对一个场进行操作，操作的时c候需要这个场另外某点的值，最好创建新的数组进行temp
ti.init(arch=ti.gpu)
# 读取obj文件，获取顶点信息和三角面信息。
_vertices, _indices, _normals, _normals_indices = read_obj.read_obj("bear.obj")
_vertices *= 0.4
_vertices[:, 0] += 0.3
_vertices[:, 2] += 0.3
# 由于读取到的obj模型是映射到0-1，会占满屏幕，所以这里进行缩放
# 缩小三倍(0-0.33)
# 平移(0.33 - 0.67)


grid_num = 80
quality = 1
particle_num = 20000

write_ply = 1

mpm_solver = MPMSolver(particle_num, grid_num=grid_num)
# # 将三角面片信息给碰撞检测算法，并初始化。
# # 将流体表面所用到的marching cube初始化
mpm_solver.init(_vertices, _indices, _normals, _normals_indices)
# # 初始化固体level set
mpm_solver.init_solid_level_set()

mpm_solver.add_cube(ti.Vector([0.35, 0.5, 0.35]), 0.23, particle_num, 0)

# 获得的坐标点被映射到了0-1之间
res = (1920, 1080)
#
window = ti.ui.Window("Real MPM 3D", res, vsync=True)

canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
camera.position(0.5, 1.0, 1.95)
camera.lookat(0.5, 0.3, 0.5)
camera.fov(55)

draw_mesh = 1
draw_particles = 1
frame_id = 0

while frame_id < 500:
    # while frame_id < 500:
    frame_id += 1
    mpm_solver.run(frame_id, write_ply)
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.ambient_light((0, 0, 0))
    if draw_mesh:
        scene.mesh(vertices=mpm_solver.collision_detection_solver.triangle_vertices,
                   indices=mpm_solver.collision_detection_solver.triangle_indices,
                   two_sided=True, color=(0.0, 1.0, 1.0))
    if draw_particles:
        scene.particles(mpm_solver.particles.position,
                        per_vertex_color=mpm_solver.particles.color,
                        radius=0.01)
    scene.point_light(pos=(0.5, 1.5, -0.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, -1.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(1.5, -1.5, 1.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(1.5, -1.5, 0.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(-0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(-1.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))

    canvas.scene(scene)

    window.show()
# 坐标系统：      +y
#               |
#               |
#               |
#               |
#               |_______________ +x
#              /
#             /
#            /
#           /
#          /
#        +z
