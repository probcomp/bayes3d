import jax3dp3 as j
import jax.numpy as jnp


H,W = 200,400
obs = jnp.zeros((H,W,3))
rendered = jnp.ones((H,W,4))
rendered_seg = jnp.ones((H,W))

j.gaussian_mixture_image(obs, rendered, rendered_seg, jnp.array([2.0]))
j.threedp3_likelihood(obs, rendered, rendered_seg, jnp.array([1.0]), 0.1, 0.1)

from IPython import embed; embed()
