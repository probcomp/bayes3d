import jax3dp3 as j
import trimesh
import jax.numpy as jnp
import numpy as np

j.meshcat.setup_visualizer()

cloud = np.random.rand(10,3) * 0.1

j.meshcat.show_cloud("1", cloud)

mesh = j.mesh.make_voxel_mesh_from_point_cloud(cloud, 0.05)


j.meshcat.show_trimesh("2", mesh)
