import jax.numpy as jnp
import jax
import numpy as np
import functools
from functools import partial
from jax.scipy.special import logsumexp

FILTER_SIZE = 3

@functools.partial(
    jnp.vectorize,
    signature='(m)->()',
    excluded=(1,2,3,4,5,6,7,),
)
def get_probs(
    ij,
    data_xyz: jnp.ndarray,
    model_xyz: jnp.ndarray,
    filter_size: int,
    r,
    num_latent_points,
    outlier_prob,
    outlier_volume
):
    dists = data_xyz[ij[0], ij[1], :3] - jax.lax.dynamic_slice(model_xyz, (ij[0], ij[1], 0), (2*filter_size + 1, 2*filter_size + 1, 3))
    probs = (jax.scipy.stats.norm.pdf(dists, loc=0, scale=r)).prod(-1).sum() * (1.0 - outlier_prob) / num_latent_points   + outlier_prob / outlier_volume 
    return probs


def threedp3_likelihood(
    obs_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    r,
    outlier_prob,
    outlier_volume,
):
    filter_size = FILTER_SIZE
    num_latent_points = obs_xyz.shape[1] * obs_xyz.shape[0]
    rendered_xyz_padded = jax.lax.pad(rendered_xyz,  -100.0, ((filter_size,filter_size,0,),(filter_size,filter_size,0,),(0,0,0,)))
    jj, ii = jnp.meshgrid(jnp.arange(obs_xyz.shape[1]), jnp.arange(obs_xyz.shape[0]))
    indices = jnp.stack([ii,jj],axis=-1)
    probs = get_probs(indices, obs_xyz, rendered_xyz_padded, filter_size, r, num_latent_points, outlier_prob, outlier_volume)
    return jnp.log(probs).sum()

threedp3_likelihood_parallel = jax.vmap(threedp3_likelihood, in_axes=(None, 0, None, None, None))
threedp3_likelihood_parallel_jit = jax.jit(threedp3_likelihood_parallel)
threedp3_likelihood_jit = jax.jit(threedp3_likelihood)
threedp3_likelihood_with_r_parallel_jit = jax.jit(
    jax.vmap(threedp3_likelihood_parallel, in_axes=(None, None, 0, None, None)),
)



import functools
from functools import partial
@functools.partial(
    jnp.vectorize,
    signature='(m)->()',
    excluded=(1,2,3,4,),
)
def get_counts(
    ij,
    data_xyz: jnp.ndarray,
    model_xyz: jnp.ndarray,
    filter_size: int,
    r,
):
    dists = data_xyz[ij[0], ij[1], :3] - jax.lax.dynamic_slice(model_xyz, (ij[0], ij[1], 0), (2*filter_size + 1, 2*filter_size + 1, 3))
    probs = (jnp.linalg.norm(dists, axis=-1) < r).sum()
    return probs

def threedp3_counts(
    obs_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    r,
):
    filter_size = FILTER_SIZE
    num_latent_points = obs_xyz.shape[1] * obs_xyz.shape[0]
    rendered_xyz_padded = jax.lax.pad(rendered_xyz,  -100.0, ((filter_size,filter_size,0,),(filter_size,filter_size,0,),(0,0,0,)))
    jj, ii = jnp.meshgrid(jnp.arange(obs_xyz.shape[1]), jnp.arange(obs_xyz.shape[0]))
    indices = jnp.stack([ii,jj],axis=-1)
    counts = get_counts(indices, obs_xyz, rendered_xyz_padded, filter_size, r)
    return counts
