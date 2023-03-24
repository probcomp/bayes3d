import jax3dp3 as j
import jax.numpy as jnp


H,W = 200,400
obs = jnp.zeros((H,W,3))
rendered = jnp.ones((H,W,3))

j.gaussian_mixture_image(obs, rendered, 2.0)
j.threedp3_likelihood(obs, rendered, 2.0, 0.1, 0.1)

from IPython import embed; embed()
