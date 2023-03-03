import jax.numpy as jnp
import jax3dp3 as j
import trimesh
import os
import numpy as np
import pybullet_planning
import trimesh


model_dir = "/home/nishadgothoskar/models"
mesh_paths = []
model_names = j.ycb_loader.MODEL_NAMES
offset_poses = []
for name in model_names:
    mesh_path = os.path.join(model_dir,name,"textured.obj")
    _, pose = j.mesh.center_mesh(trimesh.load(mesh_path), return_pose=True)
    offset_poses.append(pose)
    mesh_paths.append(
        mesh_path
    )



rgbd, gt_ids, gt_poses, masks = j.ycb_loader.get_test_img(
    # '48', '1', "/home/nishadgothoskar/data/bop/ycbv"
    # '52', '1', "/home/nishadgothoskar/data/bop/ycbv"
    '55', '22', "/home/nishadgothoskar/data/bop/ycbv"
)

paths = []
for i in gt_ids:
    paths.append(mesh_paths[i])

intrinsics = j.Intrinsics(
    rgbd.intrinsics.height, rgbd.intrinsics.width,
    rgbd.intrinsics.fx, rgbd.intrinsics.fx,
    rgbd.intrinsics.width/2, rgbd.intrinsics.height/2,
    rgbd.intrinsics.near, rgbd.intrinsics.far
)
poses = jnp.array([
    gt_poses[i] @ j.t3d.inverse_pose(offset_poses[gt_ids[i]])
    for i in range(len(gt_ids))
])

rgb, seg, depth = j.kubric_interface.render_kubric(paths, poses, jnp.eye(4), intrinsics, scaling_factor=1.0, lighting=5.0)
seg = seg[...,0]

rgba = j.viz.add_rgba_dimension(rgb)
rgba[seg == 0,-1] = 0.0
j.get_rgb_image(rgba).save("background_transparent.png")


rgb_viz = j.get_rgb_image(rgb)
depth_viz = j.get_depth_image(depth, max=intrinsics.far)
seg_viz = j.get_depth_image(seg, max=seg.max())
j.multi_panel(
    [
        rgb_viz,
        depth_viz,
        seg_viz
    ]
).save("test_kubric.png")

renderer = j.Renderer(intrinsics)
for p in paths:
    renderer.add_mesh_from_file(p)
img = renderer.render_multiobject(gt_poses, jnp.arange(len(paths)))
depth_viz = j.get_depth_image(img[:,:,2], max=intrinsics.far).save("mine.png")


j.setup_visualizer()
j.show_cloud("1", j.t3d.unproject_depth(depth, intrinsics).reshape(-1,3),color=j.RED)
j.show_cloud("2", j.t3d.unproject_depth(img[:,:,2], intrinsics).reshape(-1,3), color=j.BLUE)

from IPython import embed; embed()
