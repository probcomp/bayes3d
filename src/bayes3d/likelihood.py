import functools

import jax
import jax.numpy as jnp


###########
@functools.partial(
    jnp.vectorize,
    signature="(m)->()",
    excluded=(
        1,
        2,
        3,
        4,
        5,
        6,
    ),
)
def gausssian_mixture_vectorize_old(
    ij,
    observed_xyz: jnp.ndarray,
    rendered_xyz_padded: jnp.ndarray,
    variance,
    outlier_prob: float,
    outlier_volume: float,
    filter_size: int,
):
    distances = observed_xyz[ij[0], ij[1], :3] - jax.lax.dynamic_slice(
        rendered_xyz_padded,
        (ij[0], ij[1], 0),
        (2 * filter_size + 1, 2 * filter_size + 1, 3),
    )
    probabilities = jax.scipy.stats.norm.logpdf(
        distances, loc=0.0, scale=jnp.sqrt(variance)
    ).sum(-1) - jnp.log(observed_xyz.shape[0] * observed_xyz.shape[1])
    return jnp.logaddexp(
        probabilities.max() + jnp.log(1.0 - outlier_prob),
        jnp.log(outlier_prob) - jnp.log(outlier_volume),
    )


def threedp3_likelihood_per_pixel_old(
    observed_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    variance,
    outlier_prob,
    outlier_volume,
    filter_size,
):
    rendered_xyz_padded = jax.lax.pad(
        rendered_xyz,
        -100.0,
        (
            (
                filter_size,
                filter_size,
                0,
            ),
            (
                filter_size,
                filter_size,
                0,
            ),
            (
                0,
                0,
                0,
            ),
        ),
    )
    jj, ii = jnp.meshgrid(
        jnp.arange(observed_xyz.shape[1]), jnp.arange(observed_xyz.shape[0])
    )
    indices = jnp.stack([ii, jj], axis=-1)
    log_probabilities = gausssian_mixture_vectorize_old(
        indices,
        observed_xyz,
        rendered_xyz_padded,
        variance,
        outlier_prob,
        outlier_volume,
        filter_size,
    )
    return log_probabilities


def threedp3_likelihood_old(
    observed_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    variance,
    outlier_prob,
    outlier_volume,
    focal_length,
    filter_size,
):
    log_probabilities_per_pixel = threedp3_likelihood_per_pixel_old(
        observed_xyz, rendered_xyz, variance, outlier_prob, outlier_volume, filter_size
    )
    return log_probabilities_per_pixel.sum()


def threedp3_likelihood(
    observed_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    variance,
    outlier_prob,
):
    distances = jnp.linalg.norm(observed_xyz - rendered_xyz, axis=-1)
    probabilities_per_pixel = (distances < variance / 2) / variance
    average_probability = probabilities_per_pixel.mean()
    return average_probability
