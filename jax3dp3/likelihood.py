import jax.numpy as jnp
import functools
from .utils import extract_2d_patches

# @functools.partial(jax.jit, static_argnames=["r", "outlier_prob"])
def neural_descriptor_likelihood(
    obs_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    r,
    outlier_prob
):
    obs_mask = obs_xyz[:,:,2] > 0.0
    rendered_mask = rendered_xyz[:,:,2] > 0.0
    num_latent_points = rendered_mask.sum()
    rendered_xyz_patches = extract_2d_patches(rendered_xyz, (5,5))
    counts = counts_per_pixel(
        obs_xyz,
        rendered_xyz_patches,
        r,
    )
    probs = 1 / (((1 - outlier_prob)/num_latent_points) * 4/3 * jnp.pi * r**3) * counts + outlier_prob
    log_probs = jnp.log(probs)
    return jnp.sum(jnp.where(obs_mask, log_probs, 0.0))

@functools.partial(
    jnp.vectorize,
    signature='(m),(h,w,m)->()',
    excluded=(2,),
)
def counts_per_pixel(
    data_xyz: jnp.ndarray,
    model_xyz: jnp.ndarray,
    r: float,
):
    """    Args:
        data_xyz (jnp.ndarray): (3,)
            3d coordinate of observed point
        model_xyz : (filter_height, filter_width, 3),
        r : float, sphere radius
        outlier_prob: float
        num_latent_points: int
    """
    distance = jnp.linalg.norm(data_xyz - model_xyz, axis=-1).ravel() # (4,4)
    return jnp.sum(distance <= r)

@functools.partial(
    jnp.vectorize,
    signature='(m),(h,w,m)->(h,w)',
    excluded=(2,),
)
def counts_per_pixel_alternate(
    model_xyz: jnp.ndarray,
    data_xyz: jnp.ndarray,
    r: float,
):
    distance = jnp.linalg.norm(data_xyz - model_xyz, axis=-1)
    return (distance <= r)


# @functools.partial(jax.jit, static_argnames=["r", "outlier_prob"])
def neural_descriptor_likelihood_alternate(
    obs_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    r,
    outlier_prob
):
    filter_shape = (5,5)
    obs_mask = obs_xyz[:,:,2] > 0.0
    rendered_mask = rendered_xyz[:,:,2] > 0.0
    num_latent_points = rendered_mask.sum()
    obs_xyz_patches = extract_2d_patches(obs_xyz, filter_shape)
    counts_unconvolved = counts_per_pixel_alternate(
        rendered_xyz,
        obs_xyz_patches,
        r
    )

    counts = jnp.zeros(obs_xyz.shape[:2])
    for i in range(filter_shape[0]):
        for j in range(filter_shape[1]):
            shift_i = filter_shape[0]//2 - i
            shift_j = filter_shape[1]//2 - j
            counts = counts + jnp.roll(counts_unconvolved[:,:,i,j], (-shift_i, -shift_j), axis=(0,1))


    probs = 1 / (((1 - outlier_prob)/num_latent_points) * 4/3 * jnp.pi * r**3) * counts + outlier_prob
    log_probs = jnp.log(probs)
    return jnp.sum(jnp.where(obs_mask, log_probs, 0.0))
