from bayes3d.transforms_3d import transform_from_axis_angle
import jax.numpy as jnp
from scipy.spatial.transform import Rotation as R
import numpy as np

vec = np.array([0.1, 0.4, -1.0])
vec = vec / np.linalg.norm(vec)
ang = 10.0

print(R.from_rotvec(np.array(vec) * ang).as_matrix())
print(transform_from_axis_angle(jnp.array(vec), ang))

print(transform_from_axis_angle(jnp.array(vec), 0.0))