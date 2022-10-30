from jax3dp3.transforms_3d import *
from jax3dp3.distributions import gaussian_vmf
import jax
import jax.numpy as jnp

key = jax.random.PRNGKey(3) 
random_rot = gaussian_vmf(key, 0.1, 100.0)[:3,:3]
quat = rotation_matrix_to_quaternion(random_rot)
recovered_rot = quaternion_to_rotation_matrix(quat)
print(jnp.sum(jnp.abs(random_rot - recovered_rot)))

forward = jax.jit(rotation_matrix_to_quaternion)
backward = jax.jit(quaternion_to_rotation_matrix)

backward(forward(random_rot))


from IPython import embed; embed()