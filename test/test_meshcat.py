import jax3dp3
import numpy as np
import jax3dp3.transforms_3d as t3d
import jax.numpy as jnp
import meshcat.geometry as g
import trimesh
import os

jax3dp3.meshcat.setup_visualizer()

cloud = np.random.rand(1000,3) * 1.0
jax3dp3.meshcat.show_cloud("c1", cloud - 1.0)
jax3dp3.meshcat.show_cloud("c2", cloud + 4.0)

pose = t3d.transform_from_pos(jnp.array([0.0, 0.0, 0.0]))
mesh = trimesh.load(os.path.join(jax3dp3.utils.get_assets_dir(), "cube.obj"))
mesh = jax3dp3.mesh.scale_mesh(mesh, 0.1)
jax3dp3.meshcat.show_trimesh("obj", mesh)
jax3dp3.meshcat.set_pose("obj", pose)

from IPython import embed; embed()