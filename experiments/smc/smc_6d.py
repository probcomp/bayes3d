import numpy as np
import jax.numpy as jnp
import jax
from jax3dp3.model import make_scoring_function
from jax3dp3.rendering import render_planes
from jax3dp3.distributions import gaussian_vmf
from jax3dp3.viz.gif import make_gif
from jax3dp3.likelihood import threedp3_likelihood
from jax3dp3.utils import (
    make_centered_grid_enumeration_3d_points,
    quaternion_to_rotation_matrix,
    depth_to_coords_in_camera
)
from jax.scipy.stats.multivariate_normal import logpdf
from jax.scipy.special import logsumexp
from jax3dp3.shape import get_cube_shape
import time
from PIL import Image
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2

h, w, fx_fy, cx_cy = (
    120,
    160,
    jnp.array([200.0, 200.0]),
    jnp.array([80.0, 60.0]),
)
r = 0.02
outlier_prob = 0.01

shape = get_cube_shape(0.5)

render_from_pose = lambda pose: render_planes(pose,shape,h,w,fx_fy,cx_cy)
render_from_pose_jit = jax.jit(render_from_pose)
render_planes_parallel_jit = jax.jit(jax.vmap(lambda x: render_from_pose(x)))



delta_pose = jnp.array([
    [1.0, 0.0, 0.0, 0.04],   
    [0.0, 1.0, 0.0, 0.03],   
    [0.0, 0.0, 1.0, 0.01],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)
rot = R.from_euler('zyx', [0.0, 5.0, 0.0], degrees=True).as_matrix()
delta_pose = delta_pose.at[:3,:3].set(jnp.array(rot))
num_frames= 50
gt_poses = [jnp.array([
    [1.0, 0.0, 0.0, -0.6],   
    [0.0, 1.0, 0.0, -0.6],   
    [0.0, 0.0, 1.0, 4.0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)]
for t in range(num_frames):
    gt_poses.append(gt_poses[-1].dot(delta_pose))
gt_poses = jnp.stack(gt_poses)
print("gt_poses.shape", gt_poses.shape)

ground_truth_images = render_planes_parallel_jit(gt_poses)
print('ground_truth_images.shape ',ground_truth_images.shape)
make_gif(ground_truth_images, 8.0, "aismc_data.gif")

def likelihood(x, obs):
    rendered_image = render_from_pose(x)
    weight = threedp3_likelihood(obs, rendered_image, r, outlier_prob)
    return weight
likelihood_parallel = jax.vmap(likelihood, in_axes = (0, None))
likelihood_parallel_jit = jax.jit(likelihood_parallel)

categorical_vmap = jax.vmap(jax.random.categorical, in_axes=(None, 0))
logsumexp_vmap = jax.vmap(logsumexp)



DRIFT_VAR = 0.01

def run_inference(initial_particles, gt_images):
    def particle_filtering_step(data, gt_image):
        particles, weights, keys = data
        drift_poses = jax.vmap(gaussian_vmf, in_axes=(0, None, None))(keys, DRIFT_VAR, 600.0)
        particles = jnp.einsum("...ij,...jk->...ik", particles, drift_poses)
        weights = weights + likelihood_parallel(particles, gt_image)
        parent_idxs = jax.random.categorical(keys[0], weights, shape=weights.shape)
        particles = particles[parent_idxs]
        weights = jnp.full(weights.shape[0],logsumexp(weights) - jnp.log(weights.shape[0]))
        keys = jax.random.split(keys[0], weights.shape[0])
        return (particles, weights, keys), particles

    initial_weights = jnp.full(initial_particles.shape[0], 0.0)
    initial_key = jax.random.PRNGKey(3)
    initial_keys = jax.random.split(initial_key, initial_particles.shape[0])
    return jax.lax.scan(particle_filtering_step, (initial_particles, initial_weights, initial_keys), gt_images)


run_inference_jit = jax.jit(run_inference)
num_particles = 500
particles = []
for _ in range(num_particles):
    particles.append(gt_poses[0])
particles = jnp.stack(particles)
_,x = run_inference_jit(particles, ground_truth_images)



start = time.time()
_,x = run_inference_jit(particles, ground_truth_images)
end = time.time()
print ("Time elapsed:", end - start)
print ("FPS:", ground_truth_images.shape[0] / (end - start))

inferred_images = render_planes_parallel_jit(x[:,-1,:,:])
make_gif(inferred_images, 8.0, "aismc_data_inferred.gif")

from IPython import embed; embed()
