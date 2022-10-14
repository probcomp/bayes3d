import numpy as np
import jax.numpy as jnp
import jax
from jax3dp3.model import make_scoring_function
from jax3dp3.rendering import render_planes
from jax3dp3.distributions import VonMisesFisher
from jax3dp3.viz.gif import make_gif
from jax3dp3.likelihood import neural_descriptor_likelihood
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
    300,
    300,
    jnp.array([200.0, 200.0]),
    jnp.array([150.0, 150.0]),
)
r = 0.2
outlier_prob = 0.01
pixel_smudge = 0

shape = get_cube_shape(0.5)

render_from_x = lambda x: render_planes(jnp.array([
    [1.0, 0.0, 0.0, x],   
    [0.0, 1.0, 0.0, 0.0],   
    [0.0, 0.0, 1.0, 2.0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
),shape,h,w,fx_fy,cx_cy)
render_from_x_jit = jax.jit(render_from_x)
render_planes_parallel_jit = jax.jit(jax.vmap(lambda x: render_from_x(x)))

ground_truth_ys = jnp.linspace(-1.0, 1.0, 40)
ground_truth_images = render_planes_parallel_jit(ground_truth_ys)
print('ground_truth_images.shape ',ground_truth_images.shape)
make_gif(ground_truth_images, 5.0, "aismc_data.gif")

def likelihood(x, obs):
    rendered_image = render_from_x(x)
    weight = neural_descriptor_likelihood(obs, rendered_image, r, outlier_prob)
    return weight
likelihood_parallel = jax.vmap(likelihood, in_axes = (0, None))
likelihood_parallel_jit = jax.jit(likelihood_parallel)
print('likelihood(-1.0, ground_truth_images[0,:,:]) ',likelihood(-1.0, ground_truth_images[0,:,:]))

categorical_vmap = jax.vmap(jax.random.categorical, in_axes=(None, 0))

logsumexp_vmap = jax.vmap(logsumexp)

DRIFT_VAR = 0.1
GRID = jnp.arange(-1.0, 1.01, 0.25)


# Vanilla Particle Filtering
def vanilla_particle_filter(num_particles, key):
    particles = jnp.full(num_particles, -1.0)
    weights = jnp.full(num_particles, 0.0)
    for t in range(1,ground_truth_images.shape[0]):
        old_particles = particles
        old_weights = weights
        particles = old_particles + DRIFT_VAR * jax.random.normal(key, shape=old_particles.shape)
        weights = old_weights +  likelihood_parallel(particles, ground_truth_images[t,:,:])
        parent_idxs = jax.random.categorical(key, weights, shape=weights.shape)
        particles = particles[parent_idxs]
        weights = jnp.full(weights.shape[0],logsumexp(weights) - jnp.log(weights.shape[0]))
    run_res = logsumexp(weights) - jnp.log(weights.shape[0])
    return run_res



def aismc_particle_filtering(num_particles, key):
    particles = jnp.full(num_particles, -1.0)
    weights = jnp.full(num_particles, 0.0)
    for t in range(1,ground_truth_images.shape[0]):
        old_particles = particles
        old_weights = weights
        aux = old_particles + DRIFT_VAR * jax.random.normal(key, shape=old_particles.shape)
        aux_plus_grid = aux[:,None] + GRID
        likelihoods = likelihood_parallel_jit(aux_plus_grid.reshape(-1), ground_truth_images[t,:,:]).reshape(aux_plus_grid.shape)
        transition_prob = logpdf((aux_plus_grid - old_particles[:,None])/DRIFT_VAR, 0.0, 1.0)
        s = likelihoods + transition_prob
        s = s - logsumexp_vmap(s)[:,None]
        j = categorical_vmap(key, s) 

        prop = aux_plus_grid[jnp.arange(num_particles),j]

        r = logpdf((prop[:, None] + GRID)/DRIFT_VAR, 0.0, 1.0)
        r = r - logsumexp_vmap(r)[:,None]
        i = categorical_vmap(key, r) 

        weights = (
            likelihood_parallel_jit(prop,  ground_truth_images[t,:,:])
            + logpdf((prop - old_particles)/DRIFT_VAR, 0.0, 1.0)
            - s[jnp.arange(num_particles),j]
            - logpdf((aux - old_particles)/DRIFT_VAR, 0.0, 1.0)
            + r[jnp.arange(num_particles),i]
        )
        weights = old_weights + weights

        parent_idxs = jax.random.categorical(key, weights, shape=weights.shape)
        particles = prop[parent_idxs]
        weights = jnp.full(weights.shape[0],logsumexp(weights) - jnp.log(weights.shape[0]))

    run_res = logsumexp(weights) - jnp.log(weights.shape[0])
    return run_res


vanilla_particle_filter_jit = jax.jit(vanilla_particle_filter, static_argnums=(0,))
aismc_particle_filtering_jit = jax.jit(aismc_particle_filtering, static_argnums=(0, ))


from IPython import embed; embed()

key = jax.random.PRNGKey(5)
num_particles = 20
val = vanilla_particle_filter(num_particles, key)
val = aismc_particle_filtering(num_particles, key)

num_particles_sweep = [2, 5, 10, 20]
values = []
for num_particles in num_particles_sweep:
    _, key = jax.random.split(key)
    sub_values = []
    for _ in range(5):
        val = vanilla_particle_filter(num_particles, key)
        sub_values.append(val)
    values.append(sub_values)

values2 = []
for num_particles in num_particles_sweep:
    _, key = jax.random.split(key)
    sub_values = []
    for _ in range(5):
        val = aismc_particle_filtering(num_particles, key)
        sub_values.append(val)
    values2.append(sub_values)

values_np = [np.mean([j.item() for j in i]) for i in values]
values_2_np = [np.mean([j.item() for j in i]) for i in values2]

import matplotlib.pyplot as plt
plt.clf()
plt.plot(num_particles_sweep, values_np, label="vanilla")
plt.plot(num_particles_sweep, values_2_np, label="smcp3")
plt.legend()
plt.savefig("imgs/test.png")

from IPython import embed; embed()






