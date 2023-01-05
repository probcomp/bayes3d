import jax.numpy as jnp
import jax3dp3
import trimesh
import os

test_img = jax3dp3.ycb_loader.get_test_img('52', '1', "/home/nishadgothoskar/data/bop/ycbv/test")
h, w = test_img.get_image_dims()

fx, fy, cx, cy = test_img.get_camera_intrinsics()
print("intrinsics:", h, w, fx, fy, cx, cy)
h, w, fx, fy, cx, cy  = jax3dp3.camera.scale_camera_parameters(h,w,fx,fy,cx,cy, 0.5)
print("intrinsics:", h, w, fx, fy, cx, cy)
near = 1.0; far = 10000.0

jax3dp3.setup_renderer(h, w, fx, fy, cx, cy, near, far)


model_dir = "/home/nishadgothoskar/data/bop/ycbv/models"
for idx in range(1,22):
    mesh = trimesh.load(os.path.join(model_dir,"obj_" + "{}".format(idx).rjust(6, '0') + ".ply"))
    jax3dp3.load_model(mesh, h, w)


cam_pose = test_img.get_camera_pose()
object_poses = test_img.get_gt_poses()
object_ids = test_img.get_gt_indices()
rgb_img = test_img.get_rgb_image()

jax3dp3.viz.save_rgb_image(rgb_img, 255.0, "rgb.png")

idx = 0
all_object_poses = jnp.array(object_poses)

gt_image = jax3dp3.render_multiobject(all_object_poses, h,w, [x-1 for x in object_ids])
jax3dp3.viz.save_depth_image(gt_image[:,:,2], "render_test.png", min=near, max=far)

for i in range(len(object_ids)):
    gt_image = jax3dp3.render_single_object(object_poses[i], h,w, object_ids[i]-1)
    jax3dp3.viz.save_depth_image(gt_image[:,:,2], "{}.png".format(i), min=near, max=far)



# save_depth_image(gt_image[:,:,2], "render_test.png", min=near, max=far)
# save_depth_image(test_img.get_depth_image(), "render_gt.png", min=near, max=far)

from IPython import embed; embed()
