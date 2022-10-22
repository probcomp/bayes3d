from tensorflow_probability.substrates import jax as tfp
import jax
import jax.numpy as jnp
from .transforms_3d import (
    quaternion_to_rotation_matrix,
)
tfd = tfp.distributions

VonMisesFisher = tfd.VonMisesFisher

def gaussian_vmf(key, var, conc):
    translation  = jax.random.multivariate_normal(key, jnp.zeros(3), jnp.eye(3) * var)
    v = VonMisesFisher(
        jnp.array([1.0, 0.0, 0.0, 0.0]), conc
    ).sample(seed=key)
    rot_matrix =  quaternion_to_rotation_matrix(v)
    return jnp.vstack(
        [jnp.hstack([rot_matrix, translation.reshape(3,1) ]), jnp.array([0.0, 0.0, 0.0, 1.0])]
    )