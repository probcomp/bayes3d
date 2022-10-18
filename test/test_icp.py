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
from jax3dp3.icp import get_nearest_neighbor, find_least_squares_transform_between_clouds, icp
import jax.numpy as jnp


h, w, fx_fy, cx_cy = (
    120,
    160,
    jnp.array([200.0, 200.0]),
    jnp.array([80.0, 60.0]),
)

pose_1 = jnp.array([
    [1.0, 0.0, 0.0, -0.5],   

    [0.0, 1.0, 0.0, -0.3],   
    [0.0, 0.0, 1.0, 2.0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)
rot = R.from_euler('zyx', [1.0, -4.0, -12.0]).as_matrix()
pose_1 = pose_1.at[:3,:3].set(jnp.array(rot))

rot = R.from_euler('zyx', [5.0, -1.1, -4.0], degrees=True).as_matrix()
delta_pose =     jnp.array([
    [1.0, 0.0, 0.0, 0.23],   
    [0.0, 1.0, 0.0, 0.05],   
    [0.0, 0.0, 1.0, 0.02],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)
delta_pose = delta_pose.at[:3,:3].set(jnp.array(rot))
gt_pose = delta_pose.dot(pose_1)

shape = get_cube_shape(0.5)

render_planes_lambda = lambda p: render_planes(p,shape,h,w,fx_fy,cx_cy)
render_planes_jit = jax.jit(render_planes_lambda)
render_planes_parallel_jit = jax.jit(jax.vmap(render_planes_lambda))

rendered_img = render_planes_jit(pose_1)
obs_img = render_planes_jit(gt_pose)
save_depth_image(rendered_img[:,:,2], 5.0, "img_1.png")
save_depth_image(obs_img[:,:,2], 5.0, "img_2.png")


def error(c1,c2):
    return jnp.sum(jnp.abs(c1-c2))

new_pose = icp(pose_1, render_planes_lambda, obs_img, 20, 1)
print(pose_1)
print(new_pose)
print(gt_pose)

icp_lambda = lambda p: icp(p, render_planes_lambda, obs_img, 20, 1)
icp_lambda_jit = jax.jit(icp_lambda)

new_pose = icp_lambda_jit(pose_1)

start = time.time()
new_pose = icp_lambda_jit(pose_1)
end = time.time()
print ("Time elapsed:", end - start)


save_depth_image(
    jnp.hstack([render_planes_jit(pose_1)[:,:,2],
    render_planes_jit(new_pose)[:,:,2] ,
    obs_img[:,:,2]]),
    5.0,
    "img_1_inferred.png"
)

icp_lambda_parallel = jax.vmap(icp_lambda)
icp_lambda_parallel_jit = jax.jit(icp_lambda_parallel)
many_poses = jnp.stack([pose_1 for _ in range(200)])

new_poses = icp_lambda_parallel_jit(many_poses)

start = time.time()
new_poses = icp_lambda_parallel_jit(many_poses)
end = time.time()
print ("Time elapsed:", end - start)


from IPython import embed; embed()
