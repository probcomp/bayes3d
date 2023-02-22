import jax.numpy as jnp
import jax3dp3 as j
import trimesh
import os

rgb, depth, cam_pose, intrinsics, gt_ids, gt_poses, masks = j.ycb_loader.get_test_img('52', '1', "/home/nishadgothoskar/data/bop/ycbv")

rgb_viz = j.get_rgb_image(rgb)
depth_viz = j.get_depth_image(depth, max=intrinsics.far)

renderer = j.Renderer(intrinsics)

model_dir = "/home/nishadgothoskar/data/bop/ycbv/models"
for idx in range(1,22):
    renderer.add_mesh_from_file(os.path.join(model_dir,"obj_" + "{}".format(idx).rjust(6, '0') + ".ply"),scaling_factor=1.0/1000.0)

images_single_object = []
for i in range(len(gt_ids)):
    gt_image = renderer.render_single_object(gt_poses[i], gt_ids[i])
    images_single_object.append(
        j.get_depth_image(gt_image[:,:,2], max=intrinsics.far)
    )

reconstruction_viz = j.get_depth_image(renderer.render_multiobject(gt_poses, gt_ids)[:,:,2], max=intrinsics.far)

j.multi_panel(
    [
        rgb_viz,
        depth_viz,
        *images_single_object,
        reconstruction_viz,
        j.overlay_image(rgb_viz, reconstruction_viz)
    ]
).save("test_ycb_loading.png")

from IPython import embed; embed()
