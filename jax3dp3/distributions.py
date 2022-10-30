from tensorflow_probability.substrates import jax as tfp
import jax
import jax.numpy as jnp
from .transforms_3d import (
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion
)

def gaussian_vmf(key, var, concentration):
    translation = tfp.distributions.MultivariateNormalDiag(jnp.zeros(3), jnp.ones(3) * var).sample(seed=key)
    quat = tfp.distributions.VonMisesFisher(
        jnp.array([1.0, 0.0, 0.0, 0.0]), concentration
    ).sample(seed=key)
    rot_matrix =  quaternion_to_rotation_matrix(quat)
    return jnp.vstack(
        [jnp.hstack([rot_matrix, translation.reshape(3,1) ]), jnp.array([0.0, 0.0, 0.0, 1.0])]
    )

def gaussian_vmf_sample(key, pose_mean, var, concentration):
    return pose_mean.dot(gaussian_vmf(key, var, concentration))

def gaussian_vmf_logpdf(pose, pose_mean, var, concentration):
    translation_prob = tfp.distributions.MultivariateNormalDiag(pose_mean[:3,3], jnp.ones(3) * var).log_prob(pose[:3,3])
    quat_mean = rotation_matrix_to_quaternion(pose_mean[:3,:3])
    quat = rotation_matrix_to_quaternion(pose[:3,:3])
    quat_prob = tfp.distributions.VonMisesFisher(
        quat_mean, concentration
    ).log_prob(quat)
    return translation_prob + quat_prob


