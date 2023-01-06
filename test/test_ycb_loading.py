import jax.numpy as jnp
import jax3dp3
import trimesh
import os

test_img = jax3dp3.ycb_loader.get_test_img('52', '1', "/home/nishadgothoskar/data/bop/ycbv/test")
orig_h, orig_w = test_img.get_image_dims()
fx, fy, cx, cy = test_img.get_camera_intrinsics()
print("intrinsics:", orig_h, orig_w, fx, fy, cx, cy)
h, w, fx, fy, cx, cy  = jax3dp3.camera.scale_camera_parameters(orig_h, orig_w ,fx,fy,cx,cy, 1.0)
print("intrinsics:", h, w, fx, fy, cx, cy)
near = 1.0; far = 10000.0

jax3dp3.setup_renderer(h, w, fx, fy, cx, cy, near, far)


model_dir = "/home/nishadgothoskar/data/bop/ycbv/models"
for idx in range(1,22):
    mesh = trimesh.load(os.path.join(model_dir,"obj_" + "{}".format(idx).rjust(6, '0') + ".ply"))
    jax3dp3.load_model(mesh)


cam_pose = test_img.get_camera_pose()
object_poses = test_img.get_gt_poses()
object_ids = test_img.get_gt_indices()
gt_depth_img = test_img.get_depth_image()
rgb_img_data = test_img.get_rgb_image()

rgb_img = jax3dp3.viz.get_rgb_image(rgb_img_data, 255.0)
rgb_img.save("rgb.png")

idx = 0
all_object_poses = jnp.array(object_poses)


for i in range(len(object_ids)):
    gt_image = jax3dp3.render_single_object(object_poses[i], object_ids[i])
    jax3dp3.viz.save_depth_image(gt_image[:,:,2], "{}.png".format(i), min=near, max=far)

rerendered_img = jax3dp3.render_multiobject(all_object_poses, [x for x in object_ids])
reconstruction = jax3dp3.viz.get_depth_image(gt_image[:,:,2], min=near, max=far)
reconstruction.save("reconstruction.png")

overlayed = jax3dp3.viz.overlay_image(rgb_img, jax3dp3.viz.resize_image(reconstruction, orig_h, orig_w), alpha=.8)
overlayed.save("overlay.png")

gt_depth = jax3dp3.viz.get_depth_image(gt_depth_img, min=near, max=far)
gt_depth.save("gt_depth.png")

from IPython import embed; embed()
