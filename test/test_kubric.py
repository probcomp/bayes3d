import jax.numpy as jnp
import bayes3d as b
import trimesh
import os
import numpy as np
import trimesh
from tqdm import tqdm
from bayes3d.rendering.photorealistic_renderers.kubric_interface import render_many

# --- creating the ycb dir from the working directory
bop_ycb_dir = os.path.join(b.utils.get_assets_dir(), "bop/ycbv")
rgbd, gt_ids, gt_poses, masks = b.utils.ycb_loader.get_test_img('52', '1', bop_ycb_dir)



mesh_paths = []
offset_poses = []
model_dir = os.path.join(b.utils.get_assets_dir(), "ycb_video_models/models")
for i in tqdm(gt_ids):
    mesh_path = os.path.join(model_dir, b.utils.ycb_loader.MODEL_NAMES[i],"textured.obj")
    _, pose = b.utils.mesh.center_mesh(trimesh.load(mesh_path), return_pose=True)
    offset_poses.append(pose)
    mesh_paths.append(
        mesh_path
    )

intrinsics = b.Intrinsics(
    rgbd.intrinsics.height, rgbd.intrinsics.width,
    rgbd.intrinsics.fx, rgbd.intrinsics.fx,
    rgbd.intrinsics.width/2, rgbd.intrinsics.height/2,
    rgbd.intrinsics.near, rgbd.intrinsics.far
)

poses = []
for i in range(len(gt_ids)):
    poses.append(
        gt_poses[i] @ b.t3d.inverse_pose(offset_poses[i])
    )
poses = jnp.array(poses)

rgbds = render_many(mesh_paths, poses[None,...], intrinsics, scaling_factor=1.0, lighting=5.0)


b.setup_renderer(intrinsics)
for path in mesh_paths:
    b.RENDERER.add_mesh_from_file(path)

img = b.RENDERER.render(gt_poses, jnp.arange(gt_poses.shape[0]))

kubri_rgb = b.get_rgb_image(rgbds[0].rgb)
kubric_depth = b.get_depth_image(rgbds[0].depth)
rerendered_depth = b.get_depth_image(img[:,:,2])
b.multi_panel([kubri_rgb, kubric_depth, rerendered_depth],labels=["kubric_rgb", "kubric_depth", "rerendered_depth"]).save("test_kubric.png")
