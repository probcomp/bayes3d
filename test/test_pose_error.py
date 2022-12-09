import sys
sys.path.append('.')

from jax3dp3.metrics import get_rot_error_from_poses, get_translation_error_from_poses
import jax
import jax.numpy as jnp
from jax3dp3.rendering import render_planes
from jax3dp3.shape import get_rectangular_prism_shape
from functools import partial

h, w, fx_fy, cx_cy = (
    100,
    100,
    jnp.array([50.0, 50.0]),
    jnp.array([50.0, 50.0]),
)

outlier_prob = 0.1
fx, fy = fx_fy
cx, cy = cx_cy   
K = jnp.array([[fx, 0, cx], [0, fy, cy], [0,0,1]])


shape_dims = jnp.array([0.5, 0.5, 0.5]) 
shape = get_rectangular_prism_shape(shape_dims) 
render_planes_lambda = lambda p: render_planes(p,shape,h,w,fx_fy,cx_cy)
render_planes_jit = jax.jit(render_planes_lambda)


get_translation_error_from_poses = partial(get_translation_error_from_poses, render_planes_jit)

###############################
###############################
x1, y1, z1 = 0.0, 0.0, 1.50
pose1 = jnp.array([
    [1.0, 0.0, 0.0, x1],   
    [0.0, 1.0, 0.0, y1],   
    [0.0, 0.0, 1.0, z1],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)
pose2 = pose1

err12 = get_rot_error_from_poses(pose1, pose2), get_translation_error_from_poses(pose1, pose2, K)
print("error should be 0 : ", err12)


x3, y3, z3 = 0.5, 0.5, 1.25
pose3 = jnp.array([
    [1.0, 0.0, 0.0, x3],   
    [0.0, 1.0, 0.0, y3],   
    [0.0, 0.0, 1.0, z3],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)
err13 = get_rot_error_from_poses(pose1, pose3), get_translation_error_from_poses(pose1, pose3, K)
print("error should only have translation err", err13)


theta = jnp.pi/5
Rx = jnp.array([
    [1.0, 0.0, 0.0, 0],   
    [0.0, jnp.cos(theta), jnp.sin(theta), 0],   
    [0.0, -jnp.sin(theta), jnp.cos(theta), 0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)
pose4 = Rx @ pose1
print("pose4=", pose4)
err14 = get_rot_error_from_poses(pose1, pose4), get_translation_error_from_poses(pose1, pose4, K)
print("error should have rotation err ~ pi/5", err14)


from IPython import embed; embed()
