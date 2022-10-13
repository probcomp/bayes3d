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

radius = shape
center = pose[:3,-1]
center_norm_square = jnp.linalg.norm(center)**2

r, c = jnp.meshgrid(jnp.arange(w), jnp.arange(h))
pixel_coords = jnp.stack([r,c],axis=-1)
pixel_coords_dir = jnp.concatenate([(pixel_coords - cx_cy) / fx_fy, jnp.ones((h,w,1))],axis=-1)
u = pixel_coords_dir / jnp.linalg.norm(pixel_coords_dir,axis=-1, keepdims=True)

u_dot_c = jnp.einsum("ijk,k->ij", u, -center)

nabla = u_dot_c**2 - (center_norm_square - radius**2)
valid = nabla > 0

d = valid * (-u_dot_c - jnp.sqrt(nabla))
points = jnp.einsum("...i,...ki", d[:,:,None], u[:,:,:,None])
points_homogeneous = jnp.concatenate([points, jnp.ones((h,w,1))],axis=-1)

save_depth_image(points_homogeneous[:,:,2], 10.0, "imgs/depth.png")


gt_image = render_sphere(pose, shape, h,w, fx_fy, cx_cy)
print('gt_image.shape ',gt_image.shape)


from IPython import embed; embed()
