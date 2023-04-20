import jax3dp3 as j
import jax.numpy as jnp
import jax

H,W = 200,400
obs = jnp.zeros((H,W,3))
rendered = jnp.ones((H,W,4))
rendered_seg = jnp.ones((H,W))

r_array = jnp.linspace(0.01, 0.1,100)
outlier_array = jnp.linspace(0.01, 0.1,100)

j.threedp3_likelihood(obs, rendered, rendered_seg, jnp.array([0.1, 0.2]), 0.1, 0.1)






likelihood_jit = jax.vmap(jax.vmap(j.threedp3_likelihood,
       in_axes=(None, None, None, 0, None)),
       in_axes=(None, None, 0, None, None)
)

likelihood_jit(obs, rendered, r_array, outlier_array, 0.1)


from IPython import embed; embed()
