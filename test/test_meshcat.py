import bayes3d
import numpy as np
import bayes3d.transforms_3d as t3d
import jax.numpy as jnp
import meshcat.geometry as g
import trimesh
import os

bayes3d.setup_visualizer()

cloud = np.random.rand(1000,3) * 1.0
bayes3d.show_cloud("c1", cloud - 1.0)
bayes3d.show_cloud("c2", cloud + 4.0)

pose = t3d.transform_from_pos(jnp.array([0.0, 0.0, 1.0]))
mesh = trimesh.load(os.path.join(bayes3d.utils.get_assets_dir(), "sample_objs/cube.obj"))
mesh = bayes3d.mesh.scale_mesh(mesh, 0.1)
bayes3d.meshcat.show_trimesh("obj", mesh)
bayes3d.meshcat.set_pose("obj", pose)

pose = t3d.transform_from_pos(jnp.array([1.0, 0.0, 1.0]))
bayes3d.meshcat.show_pose("pose", pose)

from IPython import embed; embed()