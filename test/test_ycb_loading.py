import jax.numpy as jnp
import jax3dp3
import trimesh
import os
import cv2

test_img = jax3dp3.ycb_loader.get_test_img('52', '1', f"{jax3dp3.utils.get_data_dir()}/ycbv_test")
orig_h, orig_w = test_img.get_image_dims()

fx, fy, cx, cy = test_img.get_camera_intrinsics()
print("intrinsics:", orig_h, orig_w, fx, fy, cx, cy)
h, w, fx, fy, cx, cy  = jax3dp3.camera.scale_camera_parameters(orig_h,orig_w,fx,fy,cx,cy, 0.90)
print("intrinsics:", h, w, fx, fy, cx, cy)
near = 1.0; far = 10000.0

jax3dp3.setup_renderer(h, w, fx, fy, cx, cy, near, far, num_layers = 128)


model_dir = f"{jax3dp3.utils.get_assets_dir()}/models"
for idx in range(1,22):
    mesh = trimesh.load(os.path.join(model_dir,"obj_" + f"{str(idx).rjust(6, '0')}.ply"))
    jax3dp3.load_model(mesh)


cam_pose = test_img.get_camera_pose()
object_poses = test_img.get_gt_poses()
object_ids = test_img.get_gt_indices()
rgb_img_data = test_img.get_rgb_image()
depth_img_data = test_img.get_depth_image()
masks_data = test_img.get_object_masks()


from IPython import embed; embed()
for i, mask_data in enumerate(masks_data):
    mask_img = jax3dp3.viz.get_depth_image(mask_data, min=0.0, max=255.0)
    mask_img.save(f"mask_{i}.png")



depth_img = jax3dp3.viz.get_depth_image(depth_img_data)
depth_img.save("depth.png")

rgb_img = jax3dp3.viz.get_rgb_image(rgb_img_data, 255.0)
rgb_img.save("rgb.png")

idx = 0
all_object_poses = jnp.array(object_poses)


for i in range(len(object_ids)):
    gt_image = jax3dp3.render_single_object(object_poses[i], object_ids[i]-1)
    jax3dp3.viz.save_depth_image(gt_image[:,:,2], "{}.png".format(i), min=near, max=far)

gt_image = jax3dp3.render_multiobject(all_object_poses,[x-1 for x in object_ids])
reconstruction = jax3dp3.viz.get_depth_image(gt_image[:,:,2], min=near, max=far)
reconstruction.save("reconstruction.png")

overlayed = jax3dp3.viz.overlay_image(rgb_img, jax3dp3.viz.resize_image(reconstruction, orig_h, orig_w), alpha=.8)
overlayed.save("overlay.png")





# save_depth_image(gt_image[:,:,2], "render_test.png", min=near, max=far)
# save_depth_image(test_img.get_depth_image(), "render_gt.png", min=near, max=far)

from IPython import embed; embed()