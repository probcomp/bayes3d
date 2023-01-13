import jax.numpy as jnp
import jax
import numpy as np
import functools
from functools import partial

@functools.partial(
    jnp.vectorize,
    signature='(m)->()',
    excluded=(1,2,3,4,),
)
def count_ii_jj(
    ij,
    data_xyz: jnp.ndarray,
    model_xyz: jnp.ndarray,
    r: float,
    filter_size: int
):
    t = data_xyz[ij[0], ij[1], :3] - jax.lax.dynamic_slice(model_xyz, (ij[0], ij[1], 0), (2*filter_size + 1, 2*filter_size + 1, 3))
    distance = jnp.linalg.norm(t, axis=-1).ravel() # (4,4)
    return jnp.sum(distance <= r)


def threedp3_likelihood(
    obs_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    r,
    outlier_prob,
    outlier_volume,
):
    filter_size = 3
    obs_mask = obs_xyz[:,:,2] > 0.0
    rendered_mask = rendered_xyz[:,:,2] > 0.0
    rendered_xyz_padded = jax.lax.pad(rendered_xyz,  -100.0, ((filter_size,filter_size,0,),(filter_size,filter_size,0,),(0,0,0,)))
    jj, ii = jnp.meshgrid(jnp.arange(obs_xyz.shape[1]), jnp.arange(obs_xyz.shape[0]))
    indices = jnp.stack([ii,jj],axis=-1)
    counts = count_ii_jj(indices, obs_xyz, rendered_xyz_padded, r, filter_size)
    num_latent_points = rendered_mask.sum()
    any_points = num_latent_points > 0
    probs = (
        any_points * jnp.nan_to_num(outlier_prob * (1.0 / outlier_volume) +  ((1.0 - outlier_prob) / num_latent_points  * 1.0 / (4/3 * jnp.pi * r**3) * counts ) )
        +
        (1- any_points) * (1.0 / outlier_volume + 0.0 * counts)
    )
    log_probs = jnp.log(probs)
    return jnp.sum(jnp.where(obs_mask, log_probs, 0.0))

threedp3_likelihood_parallel_jit = jax.jit(jax.vmap(threedp3_likelihood, in_axes=(None, 0, None, None, None)))


def threedp3_likelihood_get_counts(
    obs_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    r,
):
    filter_size = 3
    obs_mask = obs_xyz[:,:,2] > 0.0
    rendered_mask = rendered_xyz[:,:,2] > 0.0
    
    rendered_xyz_padded = jax.lax.pad(rendered_xyz,  -100.0, ((filter_size,filter_size,0,),(filter_size,filter_size,0,),(0,0,0,)))
    jj, ii = jnp.meshgrid(jnp.arange(obs_xyz.shape[1]), jnp.arange(obs_xyz.shape[0]))
    indices = jnp.stack([ii,jj],axis=-1)
    counts_obs = count_ii_jj(indices, obs_xyz, rendered_xyz_padded, r, filter_size)
    
    obs_xyz_padded = jax.lax.pad(obs_xyz,  -100.0, ((filter_size,filter_size,0,),(filter_size,filter_size,0,),(0,0,0,)))
    jj, ii = jnp.meshgrid(jnp.arange(rendered_xyz.shape[1]), jnp.arange(rendered_xyz.shape[0]))
    indices = jnp.stack([ii,jj],axis=-1)
    counts_rendered = count_ii_jj(indices, rendered_xyz, obs_xyz_padded, r, filter_size)
    return jnp.array([
        (obs_mask * (counts_obs > 0)).sum(), obs_mask.sum(), (rendered_mask * (counts_rendered > 0)).sum(), rendered_mask.sum()
    ])

