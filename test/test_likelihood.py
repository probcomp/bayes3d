import bayes3d as b
import jax.numpy as jnp

H = 100
W = 200
observed_xyz, rendered_xyz = jnp.ones((H, W, 3)), jnp.ones((H, W, 3))
b.threedp3_likelihood(observed_xyz, rendered_xyz, 0.007, 0.1, 0.1, 1.0, 3)
