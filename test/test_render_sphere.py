import numpy as np
import jax.numpy as jnp
import jax
from jax3dp3.rendering import render_sphere
from jax3dp3.utils import (
    make_centered_grid_enumeration_3d_points,
    quaternion_to_rotation_matrix,
    depth_to_coords_in_camera
)
from jax3dp3.shape import get_cube_shape
from jax3dp3.viz.img import save_depth_image

import matplotlib.pyplot as plt

h, w, fx_fy, cx_cy = (
    300,
    300,
    jnp.array([200.0, 200.0]),
    jnp.array([150.0, 150.0]),
)

pose = jnp.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 1.6],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
shape = 0.3


gt_image = render_sphere(pose, shape, h,w, fx_fy, cx_cy)
print('gt_image.shape ',gt_image.shape)
save_depth_image(gt_image[:,:,2], 10.0, "imgs/depth.png")


from IPython import embed; embed()
