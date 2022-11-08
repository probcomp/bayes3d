import jax.numpy as jnp
from jax3dp3.transforms_3d import quaternion_to_rotation_matrix, transform_from_pos

def axis_aligned_bounding_box(object_points):
    maxs = jnp.max(object_points,axis=0)
    mins = jnp.min(object_points,axis=0)
    dims = (maxs - mins)
    center = (maxs + mins) / 2
    return dims, transform_from_pos(center)    