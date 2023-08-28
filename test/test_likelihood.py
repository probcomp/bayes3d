import numpy as np
import jax.numpy as jnp
import jax
import bayes3d as b
import trimesh
import os
import time

H=100
W=200

b.threedp3_likelihood(jnp.ones((H,W,3)), jnp.ones((H,W,3)), 100000.0, 0.1, 0.1, 3)