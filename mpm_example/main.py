import taichi as ti

from mpm_example.mpm_solver import MPMSolver
from mpm_example.const import const

# 调试的时候使用debug=True，可以检测数组下标越界，在调试时候用
# 调试数组下标越界、断言等
# 3D情况调试一个场，最好用ggui绘制出来
# 注意并行，如果对一个场进行操作，操作的时候需要这个场另外某点的值，最好创建新的数组进行temp
ti.init(arch=ti.gpu)

grid_num = 128
quality = 1
particle_num = 100000

write_ply = 0

mpm_solver = MPMSolver(particle_num, grid_num=grid_num)
# # 将三角面片信息给碰撞检测算法，并初始化。
# # 将流体表面所用到的marching cube初始化

mpm_solver.init_bear()
mpm_solver.generate_bear(const.MATERIAL_FLUID)

res = (1920, 1080)
#
window = ti.ui.Window("Real MPM 3D", res, vsync=True)

canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
camera.position(0.5, 1.0, 1.95)
camera.lookat(0.5, 0.3, 0.5)
camera.fov(55)
frame = 0
while True:
    frame += 1
    mpm_solver.run(write_ply, frame)
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)
    scene.ambient_light((0, 0, 0))

    scene.particles(mpm_solver.particles.position,
                    per_vertex_color=mpm_solver.particles.color,
                    radius=0.01)
    # scene.particles(mpm_solver.grid_mark.grid_pos,
    #                 per_vertex_color=mpm_solver.grid_mark.grid_color,
    #                 radius=0.006)
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
