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
concentration = 100.0

key = jax.random.PRNGKey(3)
key2 = jax.random.PRNGKey(4)
a = gaussian_vmf(key, var, concentration)
b = gaussian_vmf(key2, var, concentration)

gaussian_vmf_logpdf(a,b,var,concentration)

f = jax.jit(gaussian_vmf_logpdf)
f(a,b,var,concentration)

for concentration in jnp.linspace(10.0, 1000.0, 10):
    print('concentration:');print(concentration)
    print('f(a,b,var,concentration):');print(f(a,a,var,concentration))





num_mixture_components = 5
log_weights = jnp.log(jnp.ones(5) / num_mixture_components)
pose_means = jnp.stack([a for _ in range(5)])

p = gaussian_vmf_mixture_sample(key, pose_means, log_weights, var, concentration)
gaussian_vmf_mixture_logpdf(key, p, pose_means, log_weights, var, concentration)

f = jax.jit(gaussian_vmf_mixture_sample)
f(key, pose_means, log_weights, var, concentration)

f = jax.jit(gaussian_vmf_mixture_logpdf)
f(key, p, pose_means, log_weights, var, concentration)


from IPython import embed; embed()
