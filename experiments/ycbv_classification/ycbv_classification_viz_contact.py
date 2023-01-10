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



## setup renderer
jax3dp3.setup_renderer(h, w, fx, fy, cx, cy, near, far, num_layers = 128)

# todo: transform all obj poses (rn in cam frame) by world
# cam .* obj pose

# tbl pose is at identity (xy plane)
# grid around (tbl cloud )

# when render inv(cam) * again



# 1) transform all poses i2
# 2) enum over poses in wf 
# 3) render in cam pose 


# add back centroid thing
# take pyb and take 1obj 1ycb 
# straightforward 

## load in the models
num_models = 21
model_dir = os.path.join(os.environ["YCB_DIR"], "models")
model_names = np.arange(num_models) #os.listdir(f"{jax3dp3.utils.get_assets_dir()}/ycb_downloaded_models")
print(f"model names = {model_names}")

model_box_dims = []
for idx in range(num_models):
    mesh = trimesh.load(os.path.join(model_dir,"obj_" + f"{str(idx+1).rjust(6, '0')}.ply"))  # 000001 to 000021
    model_box_dims.append(jax3dp3.utils.axis_aligned_bounding_box(mesh.vertices)[0])
    jax3dp3.load_model(mesh)
print(f"GT bbox dims = {model_box_dims[gt_ycb_idx]}")

## setup pose inf
gt_img = t3d.depth_to_point_cloud_image(cv2.resize(np.asarray(depth_data * (segmentation == gt_obj_number)), (w,h),interpolation=0), fx,fy,cx,cy)
gt_depth_img = jax3dp3.viz.get_depth_image(gt_img[:,:,2], max=far).resize((w,h))
rgb_img = jax3dp3.viz.get_rgb_image(rgb_img_data, max_val=255.0).resize((w,h))
gt_img_complement = jax3dp3.renderer.get_gt_img_complement(depth_data, segmentation, gt_obj_number, h, w, fx, fy, cx, cy)

gt_depth_img.save("gt_depth_image.png")
rgb_img.save("gt_rgb.png")
jax3dp3.viz.save_depth_image(gt_img_complement[:, :, 2], "gt_img_complement.png", max=far)

gt_image_single_object_cloud = t3d.point_cloud_image_to_points(gt_img)
table_pose = jnp.eye(4)
gt_points_in_table_frame = t3d.apply_transform(
    t3d.apply_transform(gt_image_single_object_cloud, cam_pose), 
    jnp.linalg.inv(table_pose)
)
point_seg = jax3dp3.utils.segment_point_cloud(gt_image_single_object_cloud, 10.0)
gt_points_in_table_frame = gt_points_in_table_frame[point_seg == 0]


center_x, center_y, _ = ( gt_points_in_table_frame.min(0) + gt_points_in_table_frame.max(0))/2
print('center_x,center_y:');print(center_x,center_y)


## TODO: Get poses to score
# centroid_pose = t3d.transform_from_pos(gt_pose[:3, 3])   # TODO comment this out and use centroid_pose from above
# poses_to_score = jnp.einsum("ij,ajk->aik", centroid_pose, rotation_deltas)  
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
scorer_parallel_jit = jax.jit(jax.vmap(jax3dp3.likelihood.threedp3_likelihood, in_axes=(None, 0, None, None, None)))

max_depth = far
object_indices = list(range(len(model_names)))
start= time.time()
all_scores = []
for idx in object_indices:
    pose_proposals = poses_from_contact_params_sweep(contact_params_sweep, face_params[0], face_params[1], table_dims, model_box_dims[idx], table_pose)
    # proposals = jnp.einsum("ij,ajk->aik", jnp.linalg.inv(cam_pose), pose_proposals)
    # proposals = pose_proposals
    proposals = jnp.einsum("ij,ajk->aik", jnp.linalg.inv(cam_pose), pose_proposals)  # score in camera frame
    images_unmasked = jax3dp3.render_parallel(proposals, idx)
    images = jax3dp3.renderer.get_masked_images(images_unmasked, gt_img_complement)

    weights = scorer_parallel_jit(gt_img, images, 8.0, 0.1, 10**3)
    best_pose_idx = weights.argmax()
    filename = "imgs/best_{}.png".format(model_names[idx])
    pred = jax3dp3.viz.get_depth_image(
        images[best_pose_idx,:,:,2], max=max_depth
    )
    # pred.save(filename)
    jax3dp3.viz.overlay_image(gt_depth_img, pred,alpha=0.5).save(filename)
    all_scores.append(weights[best_pose_idx])
print(model_names[np.argmax(all_scores)])
end= time.time()
print ("Time elapsed:", end - start)

print(np.array(model_names)[np.argsort(all_scores)])
print(np.array(all_scores)[np.argsort(all_scores)])



likelihood_r_range = [6.0] #[r for r in reversed(np.linspace(5, max_r,10))] + [r for r in reversed(np.linspace(1,5,10))] + [r for r in reversed(np.linspace(min_r,1,20))] 
outlier_prob_range = [0.01] 


# Get the best CAMERA FRAME pose and score for a model given r and outlier_p parameter
def get_model_best_results(model_idx, r_range, outlier_prob_range):
    model_best_results = dict()
    pose_proposals_wf = poses_from_contact_params_sweep(contact_params_sweep, face_params[0], face_params[1], table_dims, model_box_dims[model_idx], table_pose)
    poses_to_score = jnp.einsum("ij,ajk->aik", jnp.linalg.inv(cam_pose), pose_proposals_wf)  # score in camera frame

    images_unmasked = jax3dp3.render_parallel(poses_to_score, model_idx)
    images_of_model = jax3dp3.renderer.get_masked_images(images_unmasked, gt_img_complement)

    for r in r_range:
        for outlier_prob in outlier_prob_range:
            weights = scorer_parallel_jit(gt_img, images_of_model, r, outlier_prob, 1**3)
            best_pose_idx = weights.argmax()
            
            best_pose = poses_to_score[best_pose_idx]
            best_score = weights[best_pose_idx]
            print('best_score:');print(best_score)
            model_best_results[(r, outlier_prob)] = (best_pose, best_score)

    return model_best_results

d = get_model_best_results(0, likelihood_r_range, outlier_prob_range)

## Return the best model, CAMERA FRAME pose/score for parameter range 
def get_models_best_results(model_idxs, r_range, outlier_prob_range):
    start= time.time()
    all_models_best_results = []  # model[params] -> (best_pose, best_score)

    # for each model, store best pose,score results for every (r, outlier) parameter
    for model_idx in model_idxs:
        model_best_results = get_model_best_results(model_idx, r_range, outlier_prob_range)
        all_models_best_results.append(model_best_results)
    
        print(f"Processed best results for {model_idx} for all (r, outlier_p)")
    return all_models_best_results

d = get_models_best_results(np.arange(num_models), likelihood_r_range, outlier_prob_range)

###
# Viz
###

from likelihood_viz import sample_cloud_within_r, render_cloud_at_pose

sample_cloud_within_r_jit = jax.jit(sample_cloud_within_r)

all_images = []
K = jnp.array([
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0]])

## get GT point cloud image
gt_model_image = jax3dp3.renderer.render_single_object(gt_pose, gt_ycb_idx) 
gt_model_cloud = t3d.depth_to_coords_in_camera(gt_model_image[:,:,2], K)[0] # noiseless (model)
gt_img_cloud = t3d.depth_to_coords_in_camera(gt_img[:,:,2], K)[0]  # with noise (observed img)

## setup parameter ranges
max_r = 20
min_r = 0

print('likelihood_r_range:');print(likelihood_r_range)
print('outlier_prob_range:');print(outlier_prob_range)
TOP_K_VIZ = 5


## get best results
all_models_best_results = get_models_best_results(np.arange(num_models), likelihood_r_range, outlier_prob_range)  #  model[params] -> (best_pose_idx, best_score)

for likelihood_r in likelihood_r_range: # one frame in gif
    rows = []
    for outlier_prob in outlier_prob_range: # one row in gif


        params = (likelihood_r, outlier_prob)
        print(f"likelihood r ={likelihood_r}, outlier_prob = {outlier_prob}")

        # get best poses and scores for each model
        model_idxs = [i for i in range(num_models)]
        model_best_results = [model_best_results[params] for model_best_results in all_models_best_results]
        all_pred_poses = [model_best_result[0] for model_best_result in model_best_results]
        all_scores = [model_best_result[1] for model_best_result in model_best_results]
        scores_topk_idxs = np.argsort(all_scores)[-TOP_K_VIZ:][::-1]  # top k score,  descending order
        pred_ycb_idx = scores_topk_idxs[0]

        # make panels; make duplicate panel for best model
        top_model_idxs = [model_idxs[idx] for idx in scores_topk_idxs]
        top_pred_poses = [all_pred_poses[idx] for idx in top_model_idxs] 
        top_scores = [all_scores[idx] for idx in top_model_idxs]  
        
        # initialize panel img + label
        panels = [gt_depth_img]
        labels = [f"GT Image\n ycb model idx={gt_ycb_idx}\n"]

        # make panel img for each model 
        for i, (pred_pose, score) in enumerate(zip(top_pred_poses, top_pred_poses)):
            model_idx = top_model_idxs[i]
            pred_image_unmasked = jax3dp3.renderer.render_single_object(pred_pose, model_idx)  # unmasked predicted depth image
            pred_image = jax3dp3.renderer.get_single_masked_image(pred_image_unmasked, gt_img_complement)
            pred_depth_img = jax3dp3.viz.get_depth_image(pred_image[:,:,2], max=far)  # predicted depth image (as passed thru the scorer)
            panels.append(pred_depth_img)
            labels.append(f"Predicted Image \n ycb model idx={model_idx}\nr={int(likelihood_r*1000)/1000}, outlierp={outlier_prob}\n Score={int(top_scores[i]*100000)/100000}")
        
        # annotate best image
        labels[1] = "BEST " + labels[1]
        print('\t', labels[1])

        # make likelihood cloud (based on GT)
        sampled_cloud_r = sample_cloud_within_r_jit(gt_img_cloud, likelihood_r)  # new cloud
        rendered_sampled_cloud_r = render_cloud_at_pose(sampled_cloud_r, jnp.eye(4), h, w, jnp.array([fx, fy]), jnp.array([cx, cy]), pixel_smudge=1e-5)
        sampled_depth_img = jax3dp3.viz.get_depth_image(rendered_sampled_cloud_r[:, :, 2], max=far).resize((w, h))

        panels.append(sampled_depth_img)
        labels.append(f"Likelihood evaluation\nr={int(likelihood_r*1000)/1000}, outlierp={outlier_prob}")

        row = jax3dp3.viz.multi_panel(panels, labels, middle_width=10, top_border=75, fontsize=15)

        rows.append(row)

    # combine rows into one frame of gif
    dst = jax3dp3.viz.multi_panel_vertical(rows)
    all_images.append(dst)

# "pause" at last frame
for _ in range(5):
    all_images.append(dst)

    
all_images[0].save(
    fp=f"panels_{scene_id}_{img_id}_obj{gt_obj_number}_gt{gt_ycb_idx}.gif",
    format="GIF",
    append_images=all_images,
    save_all=True,
    duration=1000,
    loop=0,
)

from IPython import embed; embed()

