import jax.numpy as jnp
import jax
import numpy as np
import functools
from functools import partial
from jax.scipy.special import logsumexp


logerf  = lambda x: jnp.logaddexp(0.0, jnp.log(2) + jax.scipy.stats.norm.logcdf(x * jnp.sqrt(2)))

# def score_filter(
#     ij,
#     observed_xyz: jnp.ndarray,
#     rendered_xyz_padded: jnp.ndarray,
#     variance,
#     outlier_prob: float,
#     outlier_volume: float,
#     focal_length,
#     filter_size
# ):
#     p = observed_xyz[ij[0], ij[1], :3]
#     filter_latent = jax.lax.dynamic_slice(rendered_xyz_padded, (ij[0], ij[1], 0), (2*filter_size + 1, 2*filter_size + 1, 3))

#     half_widths = (filter_latent[:, :, 2] / focal_length) / 2.0
#     width_observed = (p[2] / focal_length)

#     delta = jnp.stack([half_widths, half_widths, jnp.zeros_like(half_widths)],axis=-1)
#     bottom_lefts = filter_latent - delta
#     top_rights =  filter_latent + delta

#     sigma = jnp.sqrt(variance)
#     C = (2*jnp.pi)**(-3/2)  * sigma**(-3)

#     z_term = - (p[2] - bottom_lefts[:,:,2])**2/sigma**2
#     x_term = jnp.log(jnp.sqrt(jnp.pi) * sigma / 2) +  jnp.logaddexp(
#                                                         logerf((p[0] - 2*bottom_lefts[:,:,0]) / sigma), 
#                                                     logerf((- p[0] + bottom_lefts[:,:,0] + top_rights[:,:,0]) / sigma))
#     y_term = jnp.log(jnp.sqrt(jnp.pi) * sigma / 2) + jnp.logaddexp(logerf((p[1] - 2*bottom_lefts[:,:,1]) / sigma),logerf((- p[1] + bottom_lefts[:,:,1] + top_rights[:,:,1]) / sigma))

#     log_probabilities_per_latent = x_term + y_term + z_term - jnp.log(((2*half_widths)**2).sum()) + 2*jnp.log(width_observed)
#     return log_probabilities_per_latent


def compute_score(latent, p, focal_length, variance):
    half_width = (latent[2] / focal_length) / 2.0
    width_observed = (p[2] / focal_length)

    delta = jnp.array([half_width, half_width, 0.0])
    bottom_left = latent - delta
    top_right =  latent + delta

    sigma = jnp.sqrt(variance)
    C = (2*jnp.pi)**(-3/2)  * sigma**(-3)

    x_term = jnp.log(2) + logsumexp(
        jnp.array([
            jax.scipy.stats.norm.logcdf((p[0] - bottom_left[0]) / sigma ),
            jax.scipy.stats.norm.logcdf((p[0] - top_right[0]) / sigma )
        ]),
        b=jnp.array([1.0, -1.0])
    )

    y_term = jnp.log(2) + logsumexp(
        jnp.array([
            jax.scipy.stats.norm.logcdf((p[1] - bottom_left[1]) / sigma ),
            jax.scipy.stats.norm.logcdf((p[1] - top_right[1]) / sigma )
        ]),
        b=jnp.array([1.0, -1.0])
    )

    z_term = -(p[2] - bottom_left[2])**2/sigma**2
    return x_term + y_term + z_term

@functools.partial(
    jnp.vectorize,
    signature='(m)->()',
    excluded=(1,2,3,),
)
def compute_score_vectorize(latent, p, focal_length, variance):
    return compute_score(latent, p, focal_length, variance)

def convolutional_filter(
    ij,
    observed_xyz: jnp.ndarray,
    rendered_xyz_padded: jnp.ndarray,
    variance,
    outlier_prob: float,
    outlier_volume: float,
    focal_length,
    filter_size: int,
):
    p = observed_xyz[ij[0], ij[1], :3]

    filter_latent = jax.lax.dynamic_slice(
        rendered_xyz_padded,
        (ij[0], ij[1], 0),
        (2*filter_size + 1, 2*filter_size + 1, 3)
    )
    half_widths = (filter_latent[:,:,2] / focal_length) / 2.0

    probabilities = compute_score_vectorize(filter_latent, p, focal_length, variance)
    probabilities_normalized = probabilities + jnp.log(((2*half_widths)**2)) 
    # - jnp.log(((2*half_widths)**2).sum())
    return probabilities_normalized.max()
    # return jnp.logaddexp(logsumexp(probabilities_normalized) + jnp.log(1.0 - outlier_prob), jnp.log(outlier_prob) - jnp.log(outlier_volume))

@functools.partial(
    jnp.vectorize,
    signature='(m)->()',
    excluded=(1,2,3,4,5,6,7,),
)
def convolutional_filter_vectorize(
    ij,
    observed_xyz: jnp.ndarray,
    rendered_xyz_padded: jnp.ndarray,
    variance,
    outlier_prob: float,
    outlier_volume: float,
    focal_length,
    filter_size: int,
):
    return convolutional_filter(
        ij,
        observed_xyz,
        rendered_xyz_padded,
        variance,
        outlier_prob,
        outlier_volume,
        focal_length,
        filter_size,
    )

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
    width_observed = (observed_xyz[:,:,2] / focal_length)

    log_probabilities = convolutional_filter_vectorize(
        indices, observed_xyz,
        rendered_xyz_padded,
        variance, outlier_prob, outlier_volume, 
        focal_length,
        filter_size
    ) + jnp.log(width_observed**2) - jnp.log(((width_observed)**2).sum())
    return log_probabilities

threedp3_likelihood_per_pixel_jit = jax.jit(threedp3_likelihood_per_pixel, static_argnames=('filter_size',))

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
threedp3_likelihood_parallel = jax.vmap(threedp3_likelihood, in_axes=(None, 0, None, None, None, None))
threedp3_likelihood_parallel_jit = jax.jit(jax.vmap(
    threedp3_likelihood, in_axes=(None, 0, None, None, None, None))
    ,static_argnames=('filter_size',)
)

threedp3_likelihood_per_pixel_jit = jax.jit(threedp3_likelihood_per_pixel, static_argnames=('filter_size',))
threedp3_likelihood_full_hierarchical_bayes_jit = jax.jit(jax.vmap(jax.vmap(jax.vmap(
        threedp3_likelihood,
       in_axes=(None, None, None, 0, None, None)),
       in_axes=(None, None, 0, None, None, None)),
       in_axes=(None, 0, None, None, None, None)
), static_argnames=('filter_size',))


def get_latent_filter(
    ij,
    observed_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    filter_size
):
    rendered_xyz_padded = jax.lax.pad(rendered_xyz,  -100.0, ((filter_size,filter_size,0,),(filter_size,filter_size,0,),(0,0,0,)))
    filter_latent = jax.lax.dynamic_slice(rendered_xyz_padded, (ij[0], ij[1], 0), (2*filter_size + 1, 2*filter_size + 1, 3))
    return filter_latent
