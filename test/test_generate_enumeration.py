import numpy as np
import jax.numpy as jnp
import jax
from jax3dp3.enumerations import *
from jax3dp3.transforms_3d import transform_from_pos

enumeration_grid = make_grid_enumeration(-2.0, -2.0, -2.0, 2.0, 2.0, 2.0, 5, 5, 5, 10, 10)


from IPython import embed; embed()