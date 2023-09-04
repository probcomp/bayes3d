import numpy as np
import jax.numpy as jnp
import jax
import bayes3d as b
import trimesh
import os
import time

H=100
W=200
observed_xyz, rendered_xyz = jnp.ones((H,W,3)), jnp.ones((H,W,3)) 
b.threedp3_likelihood(observed_xyz, rendered_xyz, 0.007, 0.1, 0.1, 1.0, 3)




b.get_latent_filter(jnp.array([0,0]), observed_xyz, rendered_xyz,3)

b.get_filter_scores(jnp.array([0,0]),observed_xyz, rendered_xyz, 0.1, 0.1, 0.1, 1.0, 3)
