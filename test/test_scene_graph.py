from jax3dp3.scene_graph import get_contact_planes, get_contact_transform
from jax3dp3.transforms_3d import transform_from_axis_angle
import jax.numpy as jnp
from jax3dp3.shape import get_rectangular_prism_shape

contact_planes = get_contact_planes(jnp.array([1.0, 2.0, 3.0]))
print(contact_planes)

contact_transform  = get_contact_transform(1.0, 2.0, 0.1)
print(contact_transform)

dimensions = jnp.array([
    [1.0, 2.0, 3.0],
    [3.0, 6.0, 3.0],
    [5.0, 2.0, 2.0],
])

parents = jnp.array([-1, 1, 2])
poses = jnp.stack([jnp.eye(4) for _ in range(dimensions.shape[0])])



from IPython import embed; embed()
