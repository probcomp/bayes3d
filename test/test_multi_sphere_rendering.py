import numpy as np
import jax.numpy as jnp
import jax
from jax3dp3.rendering import render_sphere, render_planes_multiobject, render_spheres
from jax3dp3.utils import (
    make_centered_grid_enumeration_3d_points,
    depth_to_coords_in_camera
)
from jax3dp3.transforms_3d import quaternion_to_rotation_matrix
from jax3dp3.shape import get_cube_shape
from jax3dp3.viz import save_depth_image
from jax3dp3.shape import get_cube_shape, get_rectangular_prism_shape
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt

h, w, fx_fy, cx_cy = (
    300,
    300,
    jnp.array([200.0, 200.0]),
    jnp.array([150.0, 150.0]),
)

pose_1 = jnp.array([
    [1.0, 0.0, 0.0, -1.0],   
    [0.0, 1.0, 0.0, -1.0],   
    [0.0, 0.0, 1.0, 5.0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)
rot = R.from_euler('zyx', [1.0, -0.1, -2.0]).as_matrix()
pose_1 = pose_1.at[:3,:3].set(jnp.array(rot))

pose_2 = jnp.array([
    [1.0, 0.0, 0.0, 1.0],   
    [0.0, 1.0, 0.0, 1.0],   
    [0.0, 0.0, 1.0, 7.0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)
rot = R.from_euler('zyx', [0.2, -0.4, 1.0]).as_matrix()
pose_2 = pose_2.at[:3,:3].set(jnp.array(rot))

poses = jnp.stack([pose_1, pose_2])
radii = jnp.array([1.3, 1.2])

depth_img = render_spheres(poses, radii, h,w,fx_fy, cx_cy)
save_depth_image(depth_img[:,:,2], 10.0, "spheres.png")


from IPython import embed; embed()
