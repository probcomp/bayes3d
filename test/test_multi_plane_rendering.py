import numpy as np
import jax.numpy as jnp
import jax
from jax3dp3.rendering import render_sphere, render_planes_multiobject
from jax3dp3.utils import (
    make_centered_grid_enumeration_3d_points,
    depth_to_coords_in_camera
)
from jax3dp3.transforms_3d import quaternion_to_rotation_matrix
from jax3dp3.shape import get_cube_shape
from jax3dp3.viz.img import save_depth_image
from jax3dp3.shape import get_cube_shape, get_rectangular_prism_shape
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt

h, w, fx_fy, cx_cy = (
    300,
    300,
    jnp.array([200.0, 200.0]),
    jnp.array([150.0, 150.0]),
)


############# THIS IS HOW YOU COMBINE SHAPES (FOR NOW) ###########
shape1_planes, shape1_dims = get_rectangular_prism_shape(jnp.array([0.4, 0.8, 0.1]))
shape2_planes, shape2_dims = get_rectangular_prism_shape(jnp.array([0.4, 0.4, 0.2]))
shape_planes = jnp.stack([shape1_planes,shape2_planes])
shape_dims = jnp.stack([shape1_dims,shape2_dims])

pose_1 = jnp.array([
    [1.0, 0.0, 0.0, -1.0],   
    [0.0, 1.0, 0.0, -1.0],   
    [0.0, 0.0, 1.0, 2.0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)
rot = R.from_euler('zyx', [1.0, -0.1, -2.0]).as_matrix()
pose_1 = pose_1.at[:3,:3].set(jnp.array(rot))

pose_2 = jnp.array([
    [1.0, 0.0, 0.0, 1.0],   
    [0.0, 1.0, 0.0, 1.0],   
    [0.0, 0.0, 1.0, 2.0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)
rot = R.from_euler('zyx', [0.2, -0.4, 1.0]).as_matrix()
pose_2 = pose_2.at[:3,:3].set(jnp.array(rot))

poses = jnp.stack([pose_1, pose_2])


gt_image = render_planes_multiobject(poses, shape_planes, shape_dims, h,w, fx_fy, cx_cy)
print('gt_image.shape ',gt_image.shape)
save_depth_image(gt_image[:,:,2], 10.0, "multiobject.png")

from IPython import embed; embed()