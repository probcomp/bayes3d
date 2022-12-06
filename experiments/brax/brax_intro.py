from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np
import brax

from PIL import Image

import open3d as o3d
import time
from brax import envs
from brax import jumpy as jp
from brax.envs import to_torch
from brax.io import html
from brax.io import image
from brax.io import mesh
import jax
from jax import numpy as jnp
import jax3dp3.transforms_3d as t3d
from jax3dp3.viz.gif import make_gif_from_pil_images
import jax3dp3

#@title A bouncy ball scene
bouncy_ball = brax.Config(dt=0.05, substeps=20, dynamics_mode='pbd')
# ground is a frozen (immovable) infinite plane


ground = bouncy_ball.bodies.add(name='ground')
ground.frozen.all = True
plane = ground.colliders.add().plane
plane.SetInParent()  # for setting an empty oneof

box_half_widths = jnp.array([0.2, 0.4, 0.9])
ball_radius = 1.0

box = bouncy_ball.bodies.add(name='ball', mass=1)
box = box.colliders.add().box
box.halfsize.x = box_half_widths[0]
box.halfsize.y = box_half_widths[1]
box.halfsize.z = box_half_widths[2]


sphere = bouncy_ball.bodies.add(name='sphere', mass=1)
sphere = sphere.colliders.add().sphere
sphere.radius = ball_radius

bouncy_ball.gravity.z = -9.8

r =     t3d.rotation_from_axis_angle(jnp.array([1.0, 1.0, 0.0]) ,jnp.pi/3).dot(
    t3d.rotation_from_axis_angle(jnp.array([0.0, 1.0, 1.0]) ,jnp.pi/3)
)
q = t3d.rotation_matrix_to_quaternion(
    r
)


qp = brax.QP(
    # position of each body in 3d (z is up, right-hand coordinates)
    pos = jnp.array([[0., 0., 0.],       # ground
                    [2., 2., 3.],
                    [2., 2., 8.],
                    ]),
    # velocity of each body in 3d
    vel = jnp.array([[0., 0., 0.],       # ground
                    [0.05, 0., 0.],
                    [0.05, 0., 0.],
                    ]),     # ball
    # rotation about center of body, as a quaternion (w, x, y, z)
    rot = jnp.array([[1., 0., 0., 0.],   # ground
                    q,
                    [1.0, 0.0, 0.0, 0.0]
                    ]), # ball
    # angular velocity about center of body in 3d
    ang = jnp.array([[0., 0., 0.],       # ground
                    [0., 0., 0.],
                    [0., 0., 0.],
                    ])      # ball
)

bouncy_ball.elasticity = 0.2 #@param { type:"slider", min: 0, max: 1.0, step:0.05 }

sys = brax.System(bouncy_ball)
stepper_jit = jax.jit(sys.step)
qp, _ = stepper_jit(qp, [])

qps = []

num_timesteps = 200
start = time.time()
for i in range(num_timesteps):
    qp, _ = stepper_jit(qp, [])
    qps.append(qp)
end = time.time()

print('FPS:');print(num_timesteps / (end-start))



###############



cam = o3d.camera.PinholeCameraParameters()
intr = o3d.camera.PinholeCameraIntrinsic(
    500, 500, 1000.0, 1000.0, 500/2 - 0.5, 500/2 -0.5
)
cam.intrinsic = intr
camera_pose = t3d.transform_from_rot_and_pos(
    t3d.rotation_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), jnp.pi/4).dot(
        t3d.rotation_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), -jnp.pi/2 - jnp.pi/4)
    ), jnp.array([17.0, -17.0, 27.0])
)
cam.extrinsic = np.array(jnp.linalg.inv(camera_pose))


vis = o3d.visualization.Visualizer()
vis.create_window(width=cam.intrinsic.width, height=cam.intrinsic.height)
view = vis.get_view_control()

def qp_to_pose(qp, i):
    return t3d.transform_from_rot_and_pos(
    t3d.quaternion_to_rotation_matrix(qp.rot[i,:])
    ,
    qp.pos[i,:]
    )

imgs = [
]

for (t,qp) in enumerate(qps):
    vis.clear_geometries()

    box_geom = o3d.geometry.TriangleMesh.create_box(
        width=box_half_widths[0] * 2.0,
        height=box_half_widths[1] * 2.0,
        depth=box_half_widths[2] * 2.0,
    )
    pose = qp_to_pose(qp, 1)
    box_geom.transform(pose.dot(t3d.transform_from_pos(-box_half_widths)))
    box_geom.paint_uniform_color([1.0, 0.0, 0.0])
    vis.add_geometry(box_geom)


    pose = qp_to_pose(qp, 2)
    sphere_geom = o3d.geometry.TriangleMesh.create_sphere(radius=ball_radius)
    sphere_geom.transform(pose)
    sphere_geom.paint_uniform_color([0.0, 1.0, 0.0])
    vis.add_geometry(sphere_geom)

    vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0))

    view.convert_from_pinhole_camera_parameters(cam)
    view.set_constant_z_far(100.0)

    vis.poll_events()
    vis.update_renderer()
    imgs.append(np.array(vis.capture_screen_float_buffer()))


pil_images = []
for x in imgs:
  max_val = 1.0
  img = Image.fromarray(
      np.rint(
          x / max_val * 255.0
      ).astype(np.int8),
      mode="RGB",
  )
  pil_images.append(img)
make_gif_from_pil_images(pil_images, "out.gif")








from IPython import embed; embed()