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
depth_data = test_img.get_depth_image()
rgb_img_data = test_img.get_rgb_image()
obj_number = 1  # 2 ok 
segmentation = test_img.get_object_masks()[obj_number]
gt_obj_idx = test_img.get_gt_indices()[obj_number]


## setup intrinsics
orig_h, orig_w = test_img.get_image_dims()
fx, fy, cx, cy = test_img.get_camera_intrinsics()
print("intrinsics:", orig_h, orig_w, fx, fy, cx, cy)
h, w, fx, fy, cx, cy  = jax3dp3.camera.scale_camera_parameters(orig_h,orig_w,fx,fy,cx,cy, 0.40)
print("intrinsics:", h, w, fx, fy, cx, cy)

near = jnp.min(depth_data[depth_data != 0]) * 0.95
far = jnp.max(depth_data) * 1.05

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
gt_img = t3d.depth_to_point_cloud_image(cv2.resize(np.asarray(depth_data * (segmentation == 255.0)), (w,h),interpolation=0), fx,fy,cx,cy)
depth_img = jax3dp3.viz.get_depth_image(gt_img[:,:,2], max=far).resize((w,h))
rgb_img = jax3dp3.viz.get_rgb_image(rgb_img_data, max_val=255.0).resize((w,h))
gt_img_complement = t3d.depth_to_point_cloud_image(cv2.resize(np.asarray(depth_data * (segmentation != 255.0)), (w,h),interpolation=0), fx,fy,cx,cy)

# segmentation = jnp.asarray(cv2.resize(np.asarray(segmentation), (w,h), interpolation=0))
depth_img.save("gt_depth_image.png")
rgb_img.save("gt_rgb.png")
jax3dp3.viz.save_depth_image(gt_img_complement[:, :, 2], "gt_img_complement.png", max=far)

non_zero_points = gt_img[gt_img[:,:,2]>0,:3]
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
    blocked = images_unmasked[:,:,:,2] >= gt_img_complement[None,:,:,2] 
    nonzero = gt_img_complement[None, :, :, 2] != 0

    images = images_unmasked * (1-(blocked * nonzero))[:,:,:, None] 

    weights = scorer_parallel_jit(images, gt_img, 0.1, 0.05)
    best_pose_idx = weights.argmax()
  
    all_scores.append(weights[best_pose_idx])

    best_model_img = jax3dp3.viz.get_depth_image(images[best_pose_idx,:,:,2], max=far)
    overlayed = jax3dp3.viz.overlay_image(depth_img, best_model_img, alpha=0.8)

    multi = jax3dp3.viz.multi_panel(
            [rgb_img, best_model_img, depth_img, overlayed],
            ["Scene", "Best pose rendered", "GT depth", "Overlay"],
            middle_width=10,
            top_border=50,
            fontsize=20
        )
    multi.save(f"imgs/best_{idx}.png")
best_model = model_names[np.argmax(all_scores)]
end= time.time()
 

gt_model_idx = gt_obj_idx
gt_mesh_name = model_names[gt_model_idx]

print("gt=", gt_mesh_name)
print("best=", best_model)
print ("Time elapsed:", end - start)

from IPython import embed; embed()