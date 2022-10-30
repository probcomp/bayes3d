from jax3dp3.transforms_3d import *
from jax3dp3.distributions import *
import jax
import jax.numpy as jnp
from tensorflow_probability.substrates import jax as tfp

key = jax.random.PRNGKey(3) 
random_pose = gaussian_vmf(key, 0.1, 100.0)[:3,:3]
random_rot = random_pose[:3,:3]
quat = rotation_matrix_to_quaternion(random_rot)
recovered_rot = quaternion_to_rotation_matrix(quat)
print(jnp.sum(jnp.abs(random_rot - recovered_rot)))

forward = jax.jit(rotation_matrix_to_quaternion)
backward = jax.jit(quaternion_to_rotation_matrix)

backward(forward(random_rot))

var = 0.1
conc = 100.0

key = jax.random.PRNGKey(3)
key2 = jax.random.PRNGKey(4)
a = gaussian_vmf(key, var, conc)
b = gaussian_vmf(key2, var, conc)

gaussian_vmf_logpdf(a,b,var,conc)

f = jax.jit(gaussian_vmf_logpdf)
f(a,b,var,conc)

for conc in jnp.linspace(10.0, 1000.0, 10):
    print('conc:');print(conc)
    print('f(a,b,var,conc):');print(f(a,a,var,conc))

from IPython import embed; embed()