import jax.numpy as jnp
import bayes3d as b
import trimesh
import os
import numpy as np
import trimesh
from tqdm import tqdm
from bayes3d.viz.open3dviz import Open3DVisualizer


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


visualizer = Open3DVisualizer(intrinsics)

visualizer.clear()
for (pose, path) in zip(poses, mesh_paths):
    visualizer.make_mesh_from_file(path, pose)
rgbd_textured_reconstruction = visualizer.capture_image(intrinsics, jnp.eye(4))

visualizer.clear()
colors = b.viz.distinct_colors(len(gt_ids))
for (i,(pose, path)) in enumerate(zip(poses, mesh_paths)):
    mesh = b.utils.load_mesh(path)
    visualizer.make_trimesh(mesh, pose, (*tuple(colors[i]), 1.0))

rgbd_color_mesh_reconstruction= visualizer.capture_image(intrinsics, jnp.eye(4))

panel = b.viz.multi_panel(
    [
        b.get_rgb_image(rgbd.rgb),
        b.get_rgb_image(rgbd_textured_reconstruction.rgb),
        b.get_rgb_image(rgbd_color_mesh_reconstruction.rgb),
        b.overlay_image(
            b.get_rgb_image(rgbd.rgb),
            b.get_rgb_image(rgbd_color_mesh_reconstruction.rgb),
        )
    ]
)

panel.save("test.png")




from IPython import embed; embed()