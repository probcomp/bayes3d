import open3d as o3d
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import matplotlib.pyplot as plt
import numpy as np
import brax
from IPython.display import HTML, Image 

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
import jax3dp3


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

widths = jnp.array([1.0, 2.0, 3.0])
box_geom = o3d.geometry.TriangleMesh.create_box(
    width=widths[0],
    height=widths[1],
    depth=widths[2],
)
# box_geom.transform(pose)
# vis.add_geometry(box_geom)
vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0))

view = vis.get_view_control()
view.convert_from_pinhole_camera_parameters(cam)
view.set_constant_z_far(100.0)

vis.poll_events()
vis.update_renderer()

vis.capture_screen_image("test.png")



from IPython import embed; embed()