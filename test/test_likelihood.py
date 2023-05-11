import bayes3d as j
import jax.numpy as jnp
import jax

H,W = 200,400
obs = jnp.zeros((H,W,3))
rendered = jnp.ones((H,W,4))
rendered_seg = jnp.ones((H,W))

r_array = jnp.linspace(0.01, 0.1,100)
outlier_array = jnp.linspace(0.01, 0.1,100)

print(j.threedp3_likelihood_multi_r_jit(
    obs, rendered, rendered_seg, jnp.array([0.6, 0.6]), 0.2, 0.1, 3
))

for i in range(10):
    print(i)
    print(j.threedp3_likelihood_jit(
        obs, rendered, 0.6, 0.2, 0.1, i
    ))

from IPython import embed; embed()
