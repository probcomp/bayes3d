import cv2
import os
import time
import trimesh
import numpy as np
import jax.numpy as jnp
import jax

import jax3dp3
import jax3dp3.transforms_3d as t3d

"""
#TODO
multipanel: GT and rendered and overlay
select min/max depths 

param settings for #1 (rotation incorrect) (r/outlier)
see gridding test.py trans/rot 
segmentation mask -> -1 and obj indices
"""


## choose a test image and object
test_img = jax3dp3.ycb_loader.get_test_img('49', '570', f"{jax3dp3.utils.get_data_dir()}/ycbv_test")
depth = test_img.get_depth_image()
obj_number = 2  # 1 doesnt work
segmentation = test_img.get_object_masks()[obj_number]
gt_obj_idx = test_img.get_gt_indices()[obj_number]


## setup intrinsics
orig_h, orig_w = test_img.get_image_dims()
fx, fy, cx, cy = test_img.get_camera_intrinsics()
print("intrinsics:", orig_h, orig_w, fx, fy, cx, cy)
h, w, fx, fy, cx, cy  = jax3dp3.camera.scale_camera_parameters(orig_h,orig_w,fx,fy,cx,cy, 0.40)
print("intrinsics:", h, w, fx, fy, cx, cy)
near = 1.0; far = max_depth = 10000.0

jax3dp3.setup_renderer(h, w, fx, fy, cx, cy, near, far, num_layers = 128)


## load in the models
num_models = 21
model_dir = f"{jax3dp3.utils.get_assets_dir()}/models"
model_names = np.arange(num_models) #os.listdir(f"{jax3dp3.utils.get_assets_dir()}/ycb_downloaded_models")
print(f"model names = {model_names}")

for idx in range(num_models):
    mesh = trimesh.load(os.path.join(model_dir,"obj_" + f"{str(idx+1).rjust(6, '0')}.ply"))  # 000001 to 000021
    jax3dp3.load_model(mesh)

## setup pose inf

## TODO: make ONE mask that has indices (and -1 otherwise)
# from IPython import embed; embed()
gt_image = t3d.depth_to_point_cloud_image(cv2.resize(np.asarray(depth * (segmentation == 255.0)), (w,h),interpolation=0), fx,fy,cx,cy)
depth_img = jax3dp3.viz.get_depth_image(gt_image[:,:,2], max=max_depth)
depth_img.save("depth_image.png")
rgb_image = test_img.get_rgb_image()
jax3dp3.viz.save_rgb_image(rgb_image, max_val=255.0, filename="gt_rgb.png")

gt_image_complement = t3d.depth_to_point_cloud_image(cv2.resize(np.asarray(depth * (segmentation != 255.0)), (w,h),interpolation=0), fx,fy,cx,cy)
jax3dp3.viz.save_depth_image(gt_image_complement[:, :, 2], "gt_img_complement.png", max=max_depth)

segmentation = jnp.asarray(cv2.resize(np.asarray(segmentation), (w,h), interpolation=0))

non_zero_points = gt_image[gt_image[:,:,2]>0,:3]
_, centroid_pose = jax3dp3.utils.axis_aligned_bounding_box(non_zero_points)
rotation_deltas = jax3dp3.enumerations.make_rotation_grid_enumeration(60, 40)

centroid_pose = t3d.transform_from_pos(test_img.get_gt_poses()[obj_number][:3, 3])
poses_to_score = jnp.einsum("ij,ajk->aik", centroid_pose, rotation_deltas)

def scorer(rendered_image, gt, r, outlier_prob):
    weight = jax3dp3.likelihood.threedp3_likelihood(gt, rendered_image, r, outlier_prob)
    return weight
scorer_parallel = jax.vmap(scorer, in_axes=(0, None, None, None))
scorer_parallel_jit = jax.jit(scorer_parallel)


## score models
start= time.time()
all_scores = []
for idx in range(num_models):
    images_unmasked = jax3dp3.render_parallel(poses_to_score, idx)
    blocked = images_unmasked[:,:,:,2] >= gt_image_complement[None,:,:,2] 
    nonzero = gt_image_complement[None, :, :, 2] != 0

    images = images_unmasked * (1-(blocked * nonzero))[:,:,:, None] 
    # images = images_unmasked * (segmentation == 255.0)[None, :, :, None]
    # images = images_unmasked

    weights = scorer_parallel_jit(images, gt_image, 0.1, 0.05)
    best_pose_idx = weights.argmax()
    best_model_img = jax3dp3.viz.get_depth_image(images[best_pose_idx,:,:,2], max=max_depth)
    overlayed = jax3dp3.viz.overlay_image(depth_img, best_model_img, alpha=0.8)
    overlayed.save(f"imgs/best_{model_names[idx]}.png")
    all_scores.append(weights[best_pose_idx])
best_model = model_names[np.argmax(all_scores)]
end= time.time()
 

gt_model_idx = gt_obj_idx
gt_mesh_name = model_names[gt_model_idx]

print("gt=", gt_mesh_name)
print("best=", best_model)
print ("Time elapsed:", end - start)

from IPython import embed; embed()