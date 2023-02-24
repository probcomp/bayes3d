import jax.numpy as jnp
import jax3dp3 as j
import trimesh
import os
import numpy as np


obj_paths = []
poses = []

rgb, depth = j.kubric.render_kubric(...)
j.get_rgb_image(rgb).save("rgb.png")


from IPython import embed; embed()
