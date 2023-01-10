import cv2
import os
import functools
import time
import trimesh
import numpy as np
import jax.numpy as jnp
import jax

import jax3dp3
import jax3dp3.transforms_3d as t3d

"""
#TODO

Karen
1. add gridding over 6 contact faces
2. factor out the masked rendering with complement image
3. get model actual names for viz

Nishad
1. Finalize table detection
2. ...


(lower priority)
see gridding test.py trans/rot 
select min/max depths for better visualization so you can see the color range

"""

## choose a test image  
scene_id = '49'     # 48 ... 59
img_id = '570'      
test_img = jax3dp3.ycb_loader.get_test_img(scene_id, img_id, os.environ["YCB_DIR"])
depth_data = test_img.get_depth_image()
rgb_img_data = test_img.get_rgb_image()

## choose gt object
gt_obj_number = 3
segmentation = test_img.get_segmentation_image() 
gt_ycb_idx = test_img.get_gt_indices()[gt_obj_number]
print("GT ycb idx=", gt_ycb_idx)

## retrieve poses
def cam_frame_to_world_frame(cam_pose, pose_cf):
    return cam_pose @ pose_cf
cam_pose = test_img.get_camera_pose()
gt_pose = test_img.get_gt_poses()[gt_obj_number]
gt_pose_wf = cam_frame_to_world_frame(cam_pose, gt_pose)
table_pose = jnp.eye(4)  # xy plane 

## setup intrinsics
orig_h, orig_w = test_img.get_image_dims()
fx, fy, cx, cy = test_img.get_camera_intrinsics()
print("intrinsics:", orig_h, orig_w, fx, fy, cx, cy)
h, w, fx, fy, cx, cy  = jax3dp3.camera.scale_camera_parameters(orig_h,orig_w,fx,fy,cx,cy, 0.25)
print("intrinsics:", h, w, fx, fy, cx, cy)
near = jnp.min(depth_data[depth_data != 0]) * 0.95
far = jnp.max(depth_data) * 1.05
jax3dp3.setup_renderer(h, w, fx, fy, cx, cy, near, far, num_layers = 128)


num_models = 21
model_dir = os.path.join(os.environ["YCB_DIR"], "models")
model_names = np.arange(num_models) #os.listdir(f"{jax3dp3.utils.get_assets_dir()}/ycb_downloaded_models")
print(f"model names = {model_names}")

model_box_dims = []
for idx in range(num_models):
    mesh = trimesh.load(os.path.join(model_dir,"obj_" + f"{str(idx+1).rjust(6, '0')}.ply"))  # 000001 to 000021
    model_box_dims.append(jax3dp3.utils.axis_aligned_bounding_box(mesh.vertices)[0])
    jax3dp3.load_model(mesh)

## Get gt images from image data
gt_img = t3d.depth_to_point_cloud_image(cv2.resize(np.asarray(depth_data * (segmentation == gt_obj_number)), (w,h),interpolation=0), fx,fy,cx,cy)
gt_depth_img = jax3dp3.viz.get_depth_image(gt_img[:,:,2], max=far).resize((w,h))
rgb_img = jax3dp3.viz.get_rgb_image(rgb_img_data, max_val=255.0).resize((w,h))
gt_img_complement = jax3dp3.renderer.get_gt_img_complement(depth_data, segmentation, gt_obj_number, h, w, fx, fy, cx, cy)

gt_depth_img.save("gt_depth_image.png")
rgb_img.save("gt_rgb.png")
jax3dp3.viz.save_depth_image(gt_img_complement[:, :, 2], "gt_img_complement.png", max=far)


center_x, center_y = -95.32071, -10.108551
print('center_x,center_y:');print(center_x,center_y)

table_face_param = 2
table_dims = jnp.array([10.0, 10.0, 1e-5])
face_params = jnp.array([table_face_param,2])

grid_width = 80.0
contact_params_sweep = jax3dp3.make_translation_grid_enumeration_3d(
    center_x-grid_width, center_y-grid_width, 0.0,
    center_x+grid_width, center_y+grid_width, jnp.pi*2,
    11, 11, 10
)
poses_from_contact_params_sweep = jax.jit(jax.vmap(jax3dp3.scene_graph.pose_from_contact, in_axes=(0, None, None, None, None, None)))


# pick an obj, get best pose(s), render masked image(s)
max_depth = far
idx = gt_ycb_idx
pose_proposals = poses_from_contact_params_sweep(contact_params_sweep, face_params[0], face_params[1], table_dims, model_box_dims[idx], table_pose)
proposals = jnp.einsum("ij,ajk->aik", jnp.linalg.inv(cam_pose), pose_proposals)  # score in camera frame
images_unmasked = jax3dp3.render_parallel(proposals, idx)

# multiple masked images
images = jax3dp3.renderer.get_masked_images(images_unmasked, gt_img_complement)
best_pose_idx = 586
unmasked = jax3dp3.viz.get_depth_image(
    images_unmasked[best_pose_idx,:,:,2], max=max_depth
)
unmasked.save("best_render_unmasked_1.png")
pred = jax3dp3.viz.get_depth_image(
    images[best_pose_idx,:,:,2], max=max_depth
)
pred.save("best_render_masked_1.png")


# single masked image
image = jax3dp3.renderer.get_single_masked_image(images_unmasked[best_pose_idx], gt_img_complement)
pred = jax3dp3.viz.get_depth_image(
    image[:,:,2], max=max_depth
)
pred.save("best_render_masked_2.png")  # should be identical to best_render_1


from IPython import embed; embed()

