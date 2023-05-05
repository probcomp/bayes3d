import jax.numpy as jnp
import jax
import numpy as np
import functools
from functools import partial
from jax.scipy.special import logsumexp

@functools.partial(
    jnp.vectorize,
    signature='(m)->()',
    excluded=(1,2,3,4,5,),
)
def gausssian_mixture_multi_r_vectorize(
    ij,
    observed_xyz: jnp.ndarray,
    rendered_xyz_padded: jnp.ndarray,
    variances_padded: jnp.ndarray,
    num_mixture_components, 
    filter_size: int,
):
    distances = (
        observed_xyz[ij[0], ij[1], :3] - 
        jax.lax.dynamic_slice(rendered_xyz_padded, (ij[0], ij[1], 0), (2*filter_size + 1, 2*filter_size + 1, 3))
    )
    variance_for_each_pixel = jax.lax.dynamic_slice(variances_padded, (ij[0], ij[1],), (2*filter_size + 1, 2*filter_size + 1,))
    probability = (
        jax.scipy.stats.norm.pdf(
            distances,
            loc=0.0,
            scale=jnp.sqrt(variance_for_each_pixel)[...,None]
        ).prod(-1) / num_mixture_components
    ).sum()
    return probability


def threedp3_likelihood_multi_r_per_pixel(
    observed_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    rendered_segmentation: jnp.ndarray,
    variance_per_object: jnp.ndarray,
    outlier_prob,
    outlier_volume,
    filter_size
):
    num_mixture_components = observed_xyz.shape[1] * observed_xyz.shape[0]

    rendered_xyz_padded = jax.lax.pad(rendered_xyz[...,:3],
        -100.0, 
        ((filter_size,filter_size,0,),(filter_size,filter_size,0,),(0,0,0,))
    )

    variances = variance_per_object[jnp.abs(rendered_segmentation[..., None] - jnp.arange(len(variance_per_object))).argmin(-1)]
    variance_padded =  jax.lax.pad(
        variances,
        1e-10, 
        ((filter_size,filter_size,0,),(filter_size,filter_size,0,),)
    )

    jj, ii = jnp.meshgrid(jnp.arange(observed_xyz.shape[1]), jnp.arange(observed_xyz.shape[0]))
    indices = jnp.stack([ii,jj],axis=-1)
    
    probabilities = gausssian_mixture_multi_r_vectorize(
        indices, observed_xyz, rendered_xyz_padded,
        variance_padded, num_mixture_components, filter_size
    )
    probabilities_with_outlier_model = probabilities * (1.0 - outlier_prob)  + outlier_prob / outlier_volume
    return jnp.log(probabilities_with_outlier_model)

threedp3_likelihood_multi_r_per_pixel_jit = jax.jit(
    threedp3_likelihood_multi_r_per_pixel, static_argnames=('filter_size',)
)

def threedp3_likelihood_multi_r(
    observed_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    rendered_segmentation,
    variance_per_object: jnp.ndarray,
    outlier_prob,
    outlier_volume,
    filter_size
):
    log_probabilities_per_pixel = threedp3_likelihood_multi_r_per_pixel(
        observed_xyz, rendered_xyz, rendered_segmentation,
        variance_per_object, outlier_prob, outlier_volume,
        filter_size
    )
    return log_probabilities_per_pixel.sum()

threedp3_likelihood_multi_r_jit = jax.jit(
    threedp3_likelihood_multi_r, static_argnames=('filter_size',)
)
threedp3_likelihood_multi_r_parallel_jit = jax.jit(
    jax.vmap(threedp3_likelihood_multi_r, in_axes=(None, 0, 0, None, None, None, None)),
    static_argnames=('filter_size',)
)
threedp3_likelihood_multi_r_full_hierarchical_bayes_jit = jax.jit(jax.vmap(jax.vmap(jax.vmap(
        threedp3_likelihood_multi_r,
       in_axes=(None, None, None, None, 0, None, None)),
       in_axes=(None, None, None, 0, None, None, None)),
       in_axes=(None, 0, 0, None, None, None, None)
), static_argnames=('filter_size',)
)



############################################################################################################


@functools.partial(
    jnp.vectorize,
    signature='(m)->()',
    excluded=(1,2,3,4,5,),
)
def gausssian_mixture_vectorize(
    ij,
    observed_xyz: jnp.ndarray,
    rendered_xyz_padded: jnp.ndarray,
    r,
    num_mixture_components, 
    filter_size: int,
):
    distances = (
        observed_xyz[ij[0], ij[1], :3] - 
        jax.lax.dynamic_slice(rendered_xyz_padded, (ij[0], ij[1], 0), (2*filter_size + 1, 2*filter_size + 1, 3))
    )
    probability = (
        jax.scipy.stats.norm.pdf(
            distances,
            loc=0.0,
            scale=jnp.sqrt(r)
        ).prod(-1) / num_mixture_components
    ).sum()
    return probability

def threedp3_likelihood_per_pixel(
    observed_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    r,
    outlier_prob,
    outlier_volume,
    filter_size
):
    num_mixture_components = observed_xyz.shape[1] * observed_xyz.shape[0]

    rendered_xyz_padded = jax.lax.pad(rendered_xyz,  -100.0, ((filter_size,filter_size,0,),(filter_size,filter_size,0,),(0,0,0,)))
    jj, ii = jnp.meshgrid(jnp.arange(observed_xyz.shape[1]), jnp.arange(observed_xyz.shape[0]))
    indices = jnp.stack([ii,jj],axis=-1)
    probabilities = gausssian_mixture_vectorize(
        indices, observed_xyz,
        rendered_xyz_padded,
        r, num_mixture_components, filter_size
    )
    probabilities_with_outlier_model = probabilities * (1.0 - outlier_prob)  + outlier_prob / outlier_volume
    return jnp.log(probabilities_with_outlier_model)

def threedp3_likelihood(
    observed_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    r,
    outlier_prob,
    outlier_volume,
    filter_size
):
    log_probabilities_per_pixel = threedp3_likelihood_per_pixel(
        observed_xyz, rendered_xyz, r,
        outlier_prob, outlier_volume, filter_size
    )
    return log_probabilities_per_pixel.sum()

threedp3_likelihood_jit = jax.jit(threedp3_likelihood,static_argnames=('filter_size',))
threedp3_likelihood_parallel_jit = jax.jit(jax.vmap(
    threedp3_likelihood, in_axes=(None, 0, None, None, None, None))
    ,static_argnames=('filter_size',)
)
threedp3_likelihood_with_r_parallel_jit = jax.jit(
    jax.vmap(jax.vmap(
        threedp3_likelihood,
        in_axes=(None, 0, None, None, None, None)),
        in_axes=(None, None, 0, None, None, None)),
)