from functools import partial
import jax
import jax.numpy as jnp


def apply_transform(coords, transform):
    coords = jnp.einsum(
        'ij,...j->...i',
        transform,
        jnp.concatenate([coords, jnp.ones(coords.shape[:-1] + (1,))], axis=-1),
    )[..., :-1]
    return coords


def render_cloud_at_pose(input_cloud, pose, h, w, fx_fy, cx_cy, pixel_smudge):
    transformed_cloud = apply_transform(input_cloud, pose)
    point_cloud = jnp.vstack([-1.0 * jnp.ones((1, 3)), transformed_cloud])

    point_cloud_normalized = point_cloud / point_cloud[:, 2].reshape(-1, 1)
    temp1 = point_cloud_normalized[:, :2] * fx_fy
    temp2 = temp1 + cx_cy
    pixels = jnp.round(temp2)

    x, y = jnp.meshgrid(jnp.arange(w), jnp.arange(h))
    matches = (jnp.abs(x[:, :, None] - pixels[:, 0]) <= pixel_smudge) & (jnp.abs(y[:, :, None] - pixels[:, 1]) <= pixel_smudge)
    matches = matches * (1000.0 - point_cloud[:,-1][None, None, :])

    a = jnp.argmax(matches, axis=-1)    

    return point_cloud[a]


def sample_coordinate_within_r(r, key, coord):
    phi = jax.random.uniform(key, minval=0.0, maxval=2*jnp.pi)
    
    new_key, subkey1, subkey2 = jax.random.split(key, 3)

    costheta = jax.random.uniform(subkey1, minval=-1.0, maxval=1.0)
    u = jax.random.uniform(subkey2, minval=0.0, maxval=1.0)

    theta = jnp.arccos(costheta)
    radius = r * jnp.cbrt(u)

    sx = radius * jnp.sin(theta)* jnp.cos(phi)
    sy = radius * jnp.sin(theta) * jnp.sin(phi)
    sz = radius * jnp.cos(theta)

    return new_key, coord + jnp.array([sx, sy, sz]) 

def sample_cloud_within_r(cloud, r):
    cloud_copy = cloud.reshape(-1, 3)  # reshape to ensure correct scan dimensions
    key = jax.random.PRNGKey(214)
    sample_coordinate_partial = partial(sample_coordinate_within_r, r)

    return jax.lax.scan(sample_coordinate_partial, key, cloud_copy)[-1]
