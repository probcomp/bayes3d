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
    excluded=(1,2,3,4,5,6,7),
)
def gausssian_mixture_per_pixel_multi_r(
    ij,
    observed_xyz: jnp.ndarray,
    model_xyz_padded: jnp.ndarray,
    r_padded: jnp.ndarray,
    num_mixture_components,
    outlier_prob,
    outlier_volume,
    filter_size: int,
):
    dists = observed_xyz[ij[0], ij[1], :3] - jax.lax.dynamic_slice(model_xyz_padded, (ij[0], ij[1], 0), (2*filter_size + 1, 2*filter_size + 1, 3))
    r = jax.lax.dynamic_slice(r_padded, (ij[0], ij[1],), (2*filter_size + 1, 2*filter_size + 1,))
    log_prob_per_mixture_component = (
        jax.scipy.stats.norm.logpdf(dists, loc=0, scale=jnp.sqrt(r)[...,None]).sum(-1) - 
        jnp.log(num_mixture_components)
    )
    log_prob = jax.scipy.special.logsumexp(
        log_prob_per_mixture_component,
        b=(1.0 - outlier_prob)
    )
    log_prob_with_outlier_model = jax.scipy.special.logsumexp(
        jnp.array([log_prob, jnp.log(outlier_prob)-jnp.log(outlier_volume)])
    )
    return log_prob_with_outlier_model

def threedp3_likelihood_multi_r(
    observed_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    rendered_seg,
    r_array,
    outlier_prob,
    outlier_volume,
):
    filter_size = FILTER_SIZE
    num_mixture_components = observed_xyz.shape[1] * observed_xyz.shape[0]

    rendered_xyz_padded = jax.lax.pad(rendered_xyz[...,:3],
        -100.0, 
        ((filter_size,filter_size,0,),(filter_size,filter_size,0,),(0,0,0,))
    )

    r = r_array[jnp.abs(rendered_seg[..., None] - jnp.arange(len(r_array))).argmin(-1)]
    r_padded =  jax.lax.pad(
        r,
        1e-10,
        ((filter_size,filter_size,0,),(filter_size,filter_size,0,),)
    )

    jj, ii = jnp.meshgrid(jnp.arange(observed_xyz.shape[1]), jnp.arange(observed_xyz.shape[0]))
    indices = jnp.stack([ii,jj],axis=-1)
    log_probs = gausssian_mixture_per_pixel_multi_r(
        indices, observed_xyz, rendered_xyz_padded,
        r_padded, num_mixture_components, 
        outlier_prob, outlier_volume, filter_size
    )
    return log_probs

threedp3_likelihood_multi_r_parallel = jax.vmap(threedp3_likelihood_multi_r, in_axes=(None, 0, 0, None, None, None))
threedp3_likelihood_multi_r_parallel_jit = jax.jit(threedp3_likelihood_multi_r_parallel)
threedp3_likelihood_multi_r_jit = jax.jit(threedp3_likelihood_multi_r)
threedp3_likelihood_multi_r_with_r_parallel_jit = jax.jit(
    jax.vmap(threedp3_likelihood_multi_r_parallel, in_axes=(None, None, None, 0, None, None)),
)


threedp3_likelihood_multi_r_full_hierarchical_bayes = jax.vmap(jax.vmap(jax.vmap(threedp3_likelihood_multi_r,
       in_axes=(None, None, None, None, 0, None)),
       in_axes=(None, None, None, 0, None, None)),
       in_axes=(None, 0, 0, None, None, None)
)
threedp3_likelihood_multi_r_full_hierarchical_bayes_jit = jax.jit(threedp3_likelihood_multi_r_full_hierarchical_bayes)



######

@functools.partial(
    jnp.vectorize,
    signature='(m)->()',
    excluded=(1,2,3,4,),
)
def gausssian_mixture_per_pixel(
    ij,
    observed_xyz: jnp.ndarray,
    model_xyz: jnp.ndarray,
    filter_size: int,
    r
):
    dists = observed_xyz[ij[0], ij[1], :3] - jax.lax.dynamic_slice(model_xyz, (ij[0], ij[1], 0), (2*filter_size + 1, 2*filter_size + 1, 3))
    probs = (jax.scipy.stats.norm.pdf(dists, loc=0, scale=jnp.sqrt(r))).prod(-1).sum()
    return probs

def gaussian_mixture_image(
    obs_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    r
):
    filter_size = FILTER_SIZE
    num_latent_points = obs_xyz.shape[1] * obs_xyz.shape[0]
    rendered_xyz_padded = jax.lax.pad(rendered_xyz,  -100.0, ((filter_size,filter_size,0,),(filter_size,filter_size,0,),(0,0,0,)))
    jj, ii = jnp.meshgrid(jnp.arange(obs_xyz.shape[1]), jnp.arange(obs_xyz.shape[0]))
    indices = jnp.stack([ii,jj],axis=-1)
    probs = gausssian_mixture_per_pixel(indices, obs_xyz, rendered_xyz_padded, filter_size, r)
    return probs

gaussian_mixture_image_jit = jax.jit(gaussian_mixture_image)



def threedp3_likelihood(
    obs_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    r,
    outlier_prob,
    outlier_volume,
):
    num_latent_points = obs_xyz.shape[1] * obs_xyz.shape[0]
    probs = gaussian_mixture_image(obs_xyz, rendered_xyz, r)
    probs_with_outlier_model = probs * (1.0 - outlier_prob) / num_latent_points   + outlier_prob / outlier_volume
    return jnp.log(probs_with_outlier_model).sum()

threedp3_likelihood_parallel = jax.vmap(threedp3_likelihood, in_axes=(None, 0, None, None, None))
threedp3_likelihood_parallel_jit = jax.jit(threedp3_likelihood_parallel)
threedp3_likelihood_jit = jax.jit(threedp3_likelihood)
threedp3_likelihood_with_r_parallel_jit = jax.jit(
    jax.vmap(threedp3_likelihood_parallel, in_axes=(None, None, 0, None, None)),
)