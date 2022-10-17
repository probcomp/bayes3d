import numpy as np
import jax.numpy as jnp
import jax
from jax3dp3.rendering import render_planes
from jax3dp3.utils import (
    make_centered_grid_enumeration_3d_points,
    quaternion_to_rotation_matrix,
    apply_transform,
    depth_to_coords_in_camera,
    transform_from_rot_and_pos
)
from jax3dp3.shape import get_cube_shape
import time
from PIL import Image
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2
from jax3dp3.shape import get_cube_shape
from jax3dp3.viz.img import save_depth_image
import jax.numpy as jnp
import functools
from jax3dp3.utils import extract_2d_patches

h, w, fx_fy, cx_cy = (
    300,
    300,
    jnp.array([200.0, 200.0]),
    jnp.array([150.0, 150.0]),
)

pose_1 = jnp.array([
    [1.0, 0.0, 0.0, -0.8],   
    [0.0, 1.0, 0.0, -1.0],   
    [0.0, 0.0, 1.0, 2.0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)
rot = R.from_euler('zyx', [10.0, -4.0, -12.0]).as_matrix()
pose_1 = pose_1.at[:3,:3].set(jnp.array(rot))

pose_2 = jnp.array([
    [1.0, 0.0, 0.0, -0.82],   
    [0.0, 1.0, 0.0, -1.02],   
    [0.0, 0.0, 1.0, 2.02],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)
rot = R.from_euler('zyx', [4.0, 2.0, 12.0]).as_matrix()
pose_2 = pose_2.at[:3,:3].set(jnp.array(rot))

shape = get_cube_shape(0.5)

render_planes_jit = jax.jit(lambda p: render_planes(p,shape,h,w,fx_fy,cx_cy))
render_planes_parallel_jit = jax.jit(jax.vmap(lambda p: render_planes(p,shape,h,w,fx_fy,cx_cy)))

img_1 = render_planes_jit(pose_1)
img_2 = render_planes_jit(pose_2)
save_depth_image(img_1[:,:,2],5.0, "img_1.png")
save_depth_image(img_2[:,:,2],5.0, "img_2.png")


def get_nearest_neighbor(
    obs_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
):
    rendered_xyz_patches = extract_2d_patches(rendered_xyz, (4,4))
    matches = find_closest_point_at_pixel(
        obs_xyz,
        rendered_xyz_patches,
    )
    return matches

@functools.partial(
    jnp.vectorize,
    signature='(m),(h,w,m)->(m)',
)
def find_closest_point_at_pixel(
    data_xyz: jnp.ndarray,
    model_xyz: jnp.ndarray,
):
    distance = jnp.linalg.norm(data_xyz - model_xyz, axis=-1)
    best_point = model_xyz[jnp.unravel_index(jnp.argmin(distance), distance.shape)]
    return best_point

def find_least_squares_transform_between_clouds(c1, c2):
    centroid1 = jnp.mean(c1, axis=0)
    centroid2 = jnp.mean(c2, axis=0)
    print(centroid1)
    print(centroid2)
    c1_centered = c1 - centroid1
    c2_centered = c2 - centroid2
    print(jnp.mean(c1_centered, axis=0))
    print(jnp.mean(c2_centered, axis=0))
    H = jnp.transpose(c1_centered).dot(c2_centered)

    print(H.shape)
    U,_,V = jnp.linalg.svd(H)
    rot = (jnp.transpose(V).dot(jnp.transpose(U)))
    if jnp.linalg.det(rot) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        modifier = jnp.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
        ])
        V_mod = modifier.dot(V)
        rot = (jnp.transpose(V).dot(jnp.transpose(U)))

    T = (centroid2 - rot.dot(centroid1))
    transform =  transform_from_rot_and_pos(rot,T)
    return transform

cap  = 3000
neighbors = get_nearest_neighbor(img_1, img_2)
ii,jj = jnp.where((neighbors[:,:,2] > 0)  * (img_1[:,:,2] > 0))

c1 = neighbors[ii[:cap], jj[:cap]][:,:3]
c2 = img_1[ii[:cap], jj[:cap]][:,:3]


key = jax.random.PRNGKey(3)
c1 = jax.random.normal(key, shape=(100,3))*100.0
rot = R.from_euler('zyx', [10.0, -20.1, -2.0], degrees=True).as_matrix()
delta_pose =     jnp.array([
    [1.0, 0.0, 0.0, 0.09],   
    [0.0, 1.0, 0.0, 0.05],   
    [0.0, 0.0, 1.0, 0.02],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)
delta_pose = delta_pose.at[:3,:3].set(jnp.array(rot))
c2 = apply_transform(c1, delta_pose)

transform = find_least_squares_transform_between_clouds(c1, c2)

print(error(c2, apply_transform(c1, transform)))

from IPython import embed; embed()
