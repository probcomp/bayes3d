import sys
sys.path.append('.')

import jax.numpy as jnp
import jax
from jax3dp3.batched_scorer import batched_scorer_parallel
from jax3dp3.likelihood import threedp3_likelihood
from jax3dp3.rendering import render_planes
from jax3dp3.shape import get_cube_shape
from jax3dp3.viz import save_depth_image, get_depth_image
import time
from functools import partial 
from jax3dp3.likelihood import threedp3_likelihood 

## Camera settings
h, w, fx_fy, cx_cy = (
    100,
    100,
    jnp.array([50.0, 50.0]),
    jnp.array([50.0, 50.0]),
)

outlier_prob = 0.1
fx, fy = fx_fy
cx, cy = cx_cy   
max_depth = 5.0
K = jnp.array([[fx, 0, cx], [0, fy, cy], [0,0,1]])

### Generate GT images
gx, gy, gz = 0.531, 0.251, 1.950
eulerx, eulery, eulerz = 0, 0, 0
gt_pose = jnp.array([
    [0.9860675,  -0.16779144, -0.04418374, gx],   
    [0.17300624,  0.92314297,  0.33919233, gy],   
    [-0.01606147, -0.34134597,  0.94141835, gz],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)
cube_length = 0.5
shape = get_cube_shape(cube_length)

render_planes_lambda = lambda p: render_planes(p,shape,h,w,fx_fy,cx_cy)
render_planes_jit = jax.jit(render_planes_lambda)
gt_image = render_planes_jit(gt_pose)


# define a parallel scorer to batch over
def scorer(pose, gt_image, r):
    rendered_image = render_planes(pose, shape, h, w, fx_fy, cx_cy)
    weight = threedp3_likelihood(gt_image, rendered_image, r, outlier_prob)
    return weight
scorer_partial = lambda pose: scorer(pose, gt_image, r)
scorer_parallel = jax.vmap(scorer_partial, in_axes = (0, ))
scorer_parallel_jit = jax.jit(scorer_parallel)

# define batched version of parallel scorer
NUM_BATCHES = 2
batched_scorer_parallel = partial(batched_scorer_parallel, scorer_parallel, NUM_BATCHES)
batched_scorer_parallel_jit = jax.jit(batched_scorer_parallel)

# define gt depth image and initial latent pose
gt_depth_img = get_depth_image(gt_image[:,:,2], max_depth)
save_depth_image(gt_image[:,:,2], "gt_img.png", max=max_depth)
latent_pose_estimate = jnp.array([
    [1.0, 0.0, 0.0, 0.5],   
    [0.0, 1.0, 0.0, 0.25],   
    [0.0, 0.0, 1.0, 2.0],   
    [0.0, 0.0, 0.0, 1.0],   
    ])
    
print("initial latent=", latent_pose_estimate)


# process proposal with einsum and run batched scorer
r = 0.5
num_proposals = 1000
translations_enums = jnp.ones((num_proposals,4,4))  # some random values
print(f"{num_proposals} proposals")

def test_batched_scorer(latent_pose_estimate, translations_enums, gt_image, r):
    proposals = jnp.einsum("...ij,...jk->...ik", latent_pose_estimate, translations_enums, optimize="optimal")
    weights = batched_scorer_parallel_jit(proposals) 
    best_pose_estimate = proposals[jnp.argmax(weights)]
    return best_pose_estimate
test_batched_scorer_jit = jax.jit(test_batched_scorer)

_ = test_batched_scorer_jit(jnp.zeros(latent_pose_estimate.shape), translations_enums, gt_image, r)

start = time.time()
_ = test_batched_scorer_jit(latent_pose_estimate, translations_enums, gt_image, r)
end = time.time()


print("Elapsed = ", end-start)

from IPython import embed; embed()
