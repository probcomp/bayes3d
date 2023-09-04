import jax.numpy as jnp
import bayes3d as b
import bayes3d.utils.ycb_loader
import trimesh
import os
from tqdm import tqdm

bop_ycb_dir = os.path.join(b.utils.get_assets_dir(), "bop/ycbv")
rgbd, gt_ids, gt_poses, masks = b.utils.ycb_loader.get_test_img('52', '1', bop_ycb_dir)

rgb_viz = b.get_rgb_image(rgbd.rgb)
depth_viz = b.get_depth_image(rgbd.depth, max=rgbd.intrinsics.far)

b.setup_renderer(rgbd.intrinsics)

model_dir =os.path.join(b.utils.get_assets_dir(), "bop/ycbv/models")
for idx in range(1,22):
    b.RENDERER.add_mesh_from_file(os.path.join(model_dir,"obj_" + "{}".format(idx).rjust(6, '0') + ".ply"),scaling_factor=1.0/1000.0)

reconstruction_viz = b.get_depth_image(b.RENDERER.render(gt_poses, gt_ids)[:,:,2], max=rgbd.intrinsics.far)

b.multi_panel(
    [
        rgb_viz,
        depth_viz,
        reconstruction_viz,
        b.overlay_image(rgb_viz, reconstruction_viz)
    ]
).save("test_ycb_loading1.png")


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

b.setup_renderer(rgbd.intrinsics)
for mesh_path in mesh_paths:
    b.RENDERER.add_mesh_from_file(mesh_path, scaling_factor=1.0)


reconstruction_viz = b.get_depth_image(b.RENDERER.render(gt_poses, jnp.arange(3))[:,:,2], max=rgbd.intrinsics.far)
b.multi_panel(
    [
        rgb_viz,
        depth_viz,
        reconstruction_viz,
        b.overlay_image(rgb_viz, reconstruction_viz)
    ]
).save("test_ycb_loading2.png")


from IPython import embed; embed()
