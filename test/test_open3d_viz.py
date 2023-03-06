import jax3dp3 as j
import jax.numpy as jnp
import numpy as np
import os
import trimesh
import copy

import open3d as o3d



intrinsics = j.Intrinsics(
    height=1000,
    width=1000,
    fx=2000.0, fy=2000.0,
    cx=400.0, cy=300.0,
    near=0.001, far=50.0
)

viz = j.o3d_viz.O3DVis(intrinsics)

viz.render.scene.clear_geometry()

pose = j.t3d.transform_from_pos(jnp.array([0.0, 0.0, 0.2]))
box = jnp.array([0.05, 0.04, 0.03])
viz.make_bounding_box(box, pose)

pose = j.t3d.transform_from_pos(jnp.array([0.0, 0.0, 0.2]))
cloud = np.random.rand(10000,3) / 20.0
moved_cloud = j.t3d.apply_transform(cloud, pose)

color = j.BLUE
colors = np.tile(color, (cloud.shape[0],1))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(moved_cloud)
pcd.colors = o3d.utility.Vector3dVector(colors)


material = o3d.visualization.rendering.MaterialRecord()
material.shader = "defaultLit"

viz.render.scene.add_geometry("pcd7", pcd, material)

viz.set_camera(intrinsics, np.eye(4))
j.get_rgb_image(viz.capture_image()).save("test_open3d_viz.png")


model_dir = "/home/nishadgothoskar/models"
mesh_paths = []
model_names = j.ycb_loader.MODEL_NAMES
offset_poses = []
for name in model_names:
    mesh_path = os.path.join(model_dir,name,"textured.obj")
    mesh_paths.append(
        mesh_path
    )


import trimesh

idx = 1
mesh = trimesh.load(mesh_paths[idx])
viz.render.scene.clear_geometry()
pose = j.t3d.transform_from_pos(jnp.array([0.0, 0.0, 2.0]))
viz.make_mesh(mesh_paths[idx], pose)
viz.render.scene.set_lighting(viz.render.scene.LightingProfile.NO_SHADOWS, (0, 0, 0))

viz.set_camera(intrinsics, np.eye(4))
j.get_rgb_image(viz.capture_image()).save("test_open3d_viz.png")



from IPython import embed; embed()