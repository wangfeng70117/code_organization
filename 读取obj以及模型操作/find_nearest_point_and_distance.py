import taichi as ti

ti.init(arch=ti.gpu)
grid_num = 128
quality = 1

# 获得的坐标点被映射到了0-1之间
triangle = ti.Vector.field(3, ti.f32, shape=3)
point = ti.Vector.field(3, ti.f32, shape=2)

res = (1920, 1080)

window = ti.ui.Window("Real MPM 3D", res, vsync=True)

frame_id = 0
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
camera.position(0.5, 1.0, 1.95)
camera.lookat(0.5, 0.5, 0.5)
camera.fov(55)


@ti.func
def point_triangle_distance_min(A, B, C, P):
    # nearest_point = ti.Vector([0.0, 0.0, 0.0])
    # min_distance = 10.0
    e0 = B - A
    e1 = C - A
    dif = A - P
    a = ti.dot(e0, e0)
    b = ti.dot(e0, e1)
    c = ti.dot(e1, e1)
    d = ti.dot(e0, dif)
    e = ti.dot(e1, dif)
    f = ti.dot(dif, dif)
    det = a * c - b * b
    sc = (b * e - c * d)
    tc = (b * d - a * e)
    # if b1 and b2 and b3:
    #     min_distance = ti.sqrt(a * sc * sc + 2 * b * sc * tc + c * tc * tc + 2 * d * sc + 2 * e * tc + f)
    #     nearest_point = A + sc * e0 + tc * e1
    if sc + tc <= det:
        if sc < 0.0:
            if tc < 0.0:
                # 4
                if d < 0.0:
                    tc = 0.0
                    sc = 1.0 if -d >= a else -d / a
                else:
                    sc = 0.0
                    temp = 1.0 if -e >= c else -e / c
                    tc = 0.0 if e >= 0.0 else temp
            else:
                sc = 0.0
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
        if sc < .0:
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
                    sc = 1 if numer >= denom else numer / denom
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
    return min_distance, nearest_point


@ti.kernel
def init():
    triangle[0] = ti.Vector([0.1, 0.3, 0.1])
    triangle[1] = ti.Vector([0.8, 0.3, 0.1])
    triangle[2] = ti.Vector([0.5, 0.3, 0.8])
    point[0] = ti.Vector([0.5, 0.5, 0.5])
    min_distance, nearest_point = point_triangle_distance_min(triangle[0], triangle[1], triangle[2], point[0])
    point[1] = nearest_point


init()


@ti.kernel
def move(direction: ti.i32):
    if direction == 0:
        point[0].z -= 0.001
    if direction == 1:
        point[0].x -= 0.001
    if direction == 2:
        point[0].z += 0.001
    if direction == 3:
        point[0].x += 0.001
    min_distance, nearest_point = point_triangle_distance_min(triangle[0], triangle[1], triangle[2], point[0])
    point[1] = nearest_point


while window.running:
    frame_id += 1
    frame_id = frame_id % 256
    if window.is_pressed(ti.ui.LEFT, 't'): move(0)
    if window.is_pressed(ti.ui.RIGHT, 'f'): move(1)
    if window.is_pressed(ti.ui.UP, 'g'): move(2)
    if window.is_pressed(ti.ui.DOWN, 'h'): move(3)
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.ambient_light((0, 0, 0))

    scene.mesh(vertices=triangle,
               two_sided=True, color=(0.0, 1.0, 1.0))
    scene.particles(point, radius=0.02)
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.5))

    canvas.scene(scene)

    window.show()
