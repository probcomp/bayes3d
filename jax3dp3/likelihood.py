import jax.numpy as jnp
import jax
import numpy as np
import functools
from functools import partial

@functools.partial(
    jnp.vectorize,
    signature='(m)->()',
    excluded=(1,2,3,4,5,6,7),
)
def probability_in_kernel(
    ij,
    data_xyz: jnp.ndarray,
    model_xyz: jnp.ndarray,
    r: float,
    num_latent_points: int,
    filter_size: int,
    outlier_prob,
    outlier_volume
):
    t = data_xyz[ij[0], ij[1], :3] - jax.lax.dynamic_slice(model_xyz, (ij[0], ij[1], 0), (2*filter_size + 1, 2*filter_size + 1, 3))
    rs = jax.lax.dynamic_slice(model_xyz, (ij[0], ij[1]), (2*filter_size + 1, 2*filter_size + 1,))
    distance = jnp.linalg.norm(t, axis=-1).ravel() # (2*filter_size + 1, 2*filter_size + 1)
    is_inlier = (distance <= rs)
    inlier_probability = is_inlier * 1.0 / (4/3 * jnp.pi * rs**3) * (1.0 - outlier_prob) / num_latent_points
    total_inlier_probability = inlier_probability.sum()
    probability = total_inlier_probability + outlier_prob * (1.0 / outlier_volume)
    return jnp.sum()

def threedp3_likelihood(
    obs_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    latent_mask,
    r_latent,
    r_outside,
    outlier_prob,
    outlier_volume,
):
    filter_size = 5

    rendered_xyz_padded = jax.lax.pad(rendered_xyz,  -100.0, ((filter_size,filter_size,0,),(filter_size,filter_size,0,),(0,0,0,)))

    r_array = latent_mask * r_latent + (1.0 - latent_mask) * r_outside
    r_array_padded = jax.lax.pad(r_array,  r_outside, ((filter_size,filter_size,0,),))

    jj, ii = jnp.meshgrid(jnp.arange(obs_xyz.shape[1]), jnp.arange(obs_xyz.shape[0]))
    indices = jnp.stack([ii,jj],axis=-1)
    num_latent_points = obs_xyz.shape[0] * obs_xyz.shape[1]

    probs = probability_in_kernel(indices, obs_xyz, rendered_xyz_padded, r_array_padded, num_latent_points, filter_size, outlier_prob, outlier_volume)    
    return jnp.log(probs).sum()

threedp3_likelihood_with_r_parallel_jit = jax.jit(
    jax.vmap(
        jax.vmap(
            jax.vmap(threedp3_likelihood, in_axes=(None, 0, 0, None, None, None, None)),
            in_axes=(None, None, None, 0, None, None, None)
        ),
        in_axes=(None, None, None, None, None, 0, None)
    )
)

# threedp3_likelihood_parallel_jit = jax.jit(jax.vmap(threedp3_likelihood, in_axes=(None, 0, None, None, None)))
# threedp3_likelihood_jit = jax.jit(threedp3_likelihood)

def pixelwise_likelihood(
    obs_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    latent_mask,
    r_latent,
    r_outside,
    outlier_prob,
    outlier_volume,
):
    r_array = latent_mask * r_latent + (1.0 - latent_mask) * r_outside

    distances = jnp.linalg.norm(rendered_xyz - obs_xyz, axis=-1)[...,None]
    probs = (distances < r_array) *  1.0 / (4/3 * jnp.pi * r_array**3) * (1.0 - outlier_prob) +  r_array * 0.0 + outlier_prob * (1 / outlier_volume)
    return jnp.log(probs).sum()

pixelwise_likelihood_parallel_jit = jax.jit(jax.vmap(pixelwise_likelihood, in_axes=(None, 0, 0, None, None, None, None)))
pixelwise_likelihood_with_r_parallel_jit = jax.jit(
    jax.vmap(
        jax.vmap(
            jax.vmap(pixelwise_likelihood, in_axes=(None, 0, 0, None, None, None, None)),
            in_axes=(None, None, None, 0, None, None, None)
        ),
        in_axes=(None, None, None, None, None, 0, None)
    )
)
pixelwise_likelihood_jit = jax.jit(pixelwise_likelihood)




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

