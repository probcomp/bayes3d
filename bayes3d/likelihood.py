import jax.numpy as jnp
import jax
import numpy as np
import functools
from functools import partial
import bayes3d as b
from jax.scipy.special import logsumexp, erf

logerf  = lambda x: jnp.logaddexp(0.0, jnp.log(2) + jax.scipy.stats.norm.logcdf(x * jnp.sqrt(2)))


def score_filter(
    ij,
    observed_xyz: jnp.ndarray,
    rendered_xyz_padded: jnp.ndarray,
    variance,
    outlier_prob: float,
    outlier_volume: float,
    focal_length,
    filter_size
):
    distances = (
        observed_xyz[ij[0], ij[1], :3] - 
        jax.lax.dynamic_slice(rendered_xyz_padded, (ij[0], ij[1], 0), (2*filter_size + 1, 2*filter_size + 1, 3))
    )
    probabilities = jax.scipy.stats.norm.logpdf(
        distances,
        loc=0.0,
        scale=jnp.sqrt(variance)
    ).sum(-1) - jnp.log(observed_xyz.shape[0] * observed_xyz.shape[1])
    return probabilities

def gaussian_mixture(
    ij,
    observed_xyz: jnp.ndarray,
    rendered_xyz_padded: jnp.ndarray,
    variance,
    outlier_prob: float,
    outlier_volume: float,
    focal_length,
    filter_size: int,
):
    probabilities = score_filter(
        ij,
        observed_xyz,
        rendered_xyz_padded,
        variance,
        outlier_prob,
        outlier_volume,
        focal_length,
        filter_size
    )
    return jnp.logaddexp(probabilities.max() + jnp.log(1.0 - outlier_prob), jnp.log(outlier_prob) - jnp.log(outlier_volume))

gaussian_mixture_vectorize = jnp.vectorize(gaussian_mixture, signature='(m)->()', excluded=(1,2,3,4,5,6,7,))

def threedp3_likelihood_per_pixel(
    observed_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    variance,
    outlier_prob,
    outlier_volume,
    focal_length,
    filter_size
):
    rendered_xyz_padded = jax.lax.pad(rendered_xyz,  -100.0, ((filter_size,filter_size,0,),(filter_size,filter_size,0,),(0,0,0,)))
    jj, ii = jnp.meshgrid(jnp.arange(observed_xyz.shape[1]), jnp.arange(observed_xyz.shape[0]))
    indices = jnp.stack([ii,jj],axis=-1)
    log_probabilities = gaussian_mixture_vectorize(
        indices, observed_xyz,
        rendered_xyz_padded,
        variance, outlier_prob, outlier_volume,
        focal_length,
        filter_size
    )
    return log_probabilities

def threedp3_likelihood(
    observed_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    variance,
    outlier_prob,
    outlier_volume,
    focal_length,
    filter_size
):
    log_probabilities_per_pixel = threedp3_likelihood_per_pixel(
        observed_xyz, rendered_xyz, variance,
        outlier_prob, outlier_volume,
        focal_length,
        filter_size
    )
    return log_probabilities_per_pixel.sum()

threedp3_likelihood_jit = jax.jit(threedp3_likelihood,static_argnames=('filter_size',))
threedp3_likelihood_per_pixel_jit = jax.jit(threedp3_likelihood_per_pixel, static_argnames=('filter_size',))

def get_latent_filter(
    ij,
    observed_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    filter_size
):
    rendered_xyz_padded = jax.lax.pad(rendered_xyz,  -100.0, ((filter_size,filter_size,0,),(filter_size,filter_size,0,),(0,0,0,)))
    filter_latent = jax.lax.dynamic_slice(rendered_xyz_padded, (ij[0], ij[1], 0), (2*filter_size + 1, 2*filter_size + 1, 3))
    return filter_latent

def get_filter_scores(
    ij,
    observed_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    variance,
    outlier_prob,
    outlier_volume,
    focal_length,
    filter_size
):
    rendered_xyz_padded = jax.lax.pad(rendered_xyz,  -100.0, ((filter_size,filter_size,0,),(filter_size,filter_size,0,),(0,0,0,)))
    log_probabilities = score_filter(
        ij,
        observed_xyz,
        rendered_xyz_padded,
        variance, outlier_prob, outlier_volume,
        focal_length,
        filter_size
    )
    return log_probabilities
