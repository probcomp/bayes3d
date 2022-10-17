import numpy as np
import jax.numpy as jnp
import jax
from jax3dp3.model import make_scoring_function
from jax3dp3.rendering import render_planes
from jax3dp3.distributions import VonMisesFisher
from jax3dp3.viz.img import save_depth_image
from jax3dp3.likelihood import neural_descriptor_likelihood
from jax3dp3.utils import (
    make_centered_grid_enumeration_3d_points,
    quaternion_to_rotation_matrix,
    depth_to_coords_in_camera
)
from jax3dp3.shape import get_cube_shape
import time
from PIL import Image
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from jax.config import config
config.update("jax_debug_nans", True)



h, w, fx_fy, cx_cy = (
    300,
    300,
    jnp.array([200.0, 200.0]),
    jnp.array([150.0, 150.0]),
)
r = 0.1
outlier_prob = 0.01
pixel_smudge = 0

shape = get_cube_shape(0.5)

num_frames = 50

gt_pose =     jnp.array([
    [1.0, 0.0, 0.0, 0.0],   
    [0.0, 1.0, 0.0, 0.0],   
    [0.0, 0.0, 1.0, 2.0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)
# rot = R.from_euler('zyx', [46.0, -3.0, -10.0], degrees=True).as_matrix()
rot = R.from_euler('zyx', [0.0, 0.0, 0.0], degrees=True).as_matrix()
rot = jnp.array(rot)
gt_pose = gt_pose.at[:3,:3].set(jnp.array(rot))

pose_left = gt_pose.copy()

render_planes_jit = jax.jit(lambda p: render_planes(p,shape,h,w,fx_fy,cx_cy))
render_planes_parallel_jit = jax.jit(jax.vmap(lambda p: render_planes(p,shape,h,w,fx_fy,cx_cy)))
gt_image = render_planes_jit(gt_pose)
save_depth_image(gt_image[:,:,2], 5.0, "imgs/depth.png")
save_depth_image(render_planes_jit(pose_left)[:,:,2], 5.0, "imgs/pose_left.png")

def test_if_problem_is_likelihood(xyz):
    pose = jnp.vstack(
        [jnp.hstack([rot, xyz.reshape(3,1) ]), jnp.array([0.0, 0.0, 0.0, 1.0])]
    )
    rendered_image = render_planes(pose, shape, h, w, fx_fy, cx_cy)
    return -jnp.sum(jnp.absolute((rendered_image - gt_image)))

print(test_if_problem_is_likelihood(jnp.array([0.0, 0.0, 2.0])))
print(test_if_problem_is_likelihood(jnp.array([0.0, 0.0, 2.1])))
print(test_if_problem_is_likelihood(gt_pose[:3,-1]))
test_if_problem_is_likelihood_jit = jax.jit(test_if_problem_is_likelihood)

test_if_problem_is_likelihood_grad = jax.grad(test_if_problem_is_likelihood)

print(test_if_problem_is_likelihood_grad(jnp.array([0.0, 0.0, 2.1])))
print(test_if_problem_is_likelihood_grad(jnp.array([0.1, 0.1, 2.1])))

test_if_problem_is_likelihood_grad_jit = jax.jit(test_if_problem_is_likelihood_grad)

# vals = []
# zs  =[]
# for z in jnp.linspace(1.5, 3.0, 1000):
#     zs.append(z)
#     vals.append(test_if_problem_is_likelihood_grad_jit(jnp.array([0.1, 0.1, z]))[-1])

vals = []
zs  =[]
for z in jnp.linspace(-1.0, 1.0, 1000):
    zs.append(z)
    vals.append(test_if_problem_is_likelihood_grad_jit(jnp.array([z, 0.0, 2.0]))[0])


plt.clf()
plt.plot(zs, vals)
plt.savefig("imgs/gradient_sweep.png")


vals = []
zs  =[]
for z in jnp.linspace(-1.0, 1.0, 1000):
    zs.append(z)
    vals.append(test_if_problem_is_likelihood_jit(jnp.array([z, 0.0, 2.0])))

plt.clf()
plt.plot(zs, vals)
plt.savefig("imgs/loss_sweep.png")

from IPython import embed; embed()