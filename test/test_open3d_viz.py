import jax3dp3 as j
import jax.numpy as jnp
import numpy as np
import os
import trimesh
import copy

import open3d as o3d


rgbd, gt_ids, gt_poses, masks = j.ycb_loader.get_test_img('52', '1', "/home/nishadgothoskar/data/bop/ycbv")

viz_intrinsics = j.Intrinsics(
    1000,1000,
    1000.0,
    1000.0,
    500.0,
    500.0,
    0.01,
    20.0
)

viz = j.o3d_viz.O3DVis(viz_intrinsics)

full_mask = jnp.array(masks).sum(0) > 0

viz.render.scene.clear_geometry()
cloud = j.t3d.apply_transform(j.t3d.unproject_depth(rgbd.depth, rgbd.intrinsics), rgbd.camera_pose)[full_mask].reshape(-1,3)
rgb_cloud = rgbd.rgb[full_mask,:3].reshape(-1,3)
viz.make_cloud(cloud, color=rgb_cloud / 255.0)

viz.make_camera(rgbd.intrinsics, rgbd.camera_pose, 0.1)

pos, target, up =(
    jnp.array([4.0, 0.0, 2.0]),
    jnp.array([0.0, 0.0, 0.0]),
    jnp.array([0.0, 0.0, 1.0]),
)
bystander_pose = j.t3d.transform_from_pos_target_up(pos, target, up)


viz.set_camera(rgbd.intrinsics, bystander_pose)
j.get_rgb_image(viz.capture_image()).save("test_open3d_viz.png")



j.get_rgb_image(rgbd.rgb).save("rgb.png")
j.get_depth_image(rgbd.depth).save("depth.png")

viz.render.scene.clear_geometry()

viz.set_camera(rgbd.intrinsics, rgbd.camera_pose)
j.get_rgb_image(viz.capture_image()).save("test_open3d_viz.png")

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