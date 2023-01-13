import numpy as np
import cv2
import jax3dp3.transforms_3d as t3d
import os
import collections
import heapq
import jax
import jax.numpy as jnp
import trimesh
import time
import jax3dp3
import pickle


scene_num = 3
file = open(f"../panda_data/panda_dataset/scene_{scene_num}.pkl",'rb')
all_data = pickle.load(file)
file.close()
t = -1
data = all_data[t]
print(data.keys())

rgb = data["rgb"]
rgb_viz = jax3dp3.viz.get_rgb_image(rgb, 255.0)
rgb_viz.save("rgb.png")
depth = data["depth"] / 1000.0
jax3dp3.viz.get_depth_image(depth, max=3000.0).save("depth.png")
K = data["intrinsics"][0]
orig_h,orig_w = depth.shape
orig_fx, orig_fy, orig_cx, orig_cy = K[0,0],K[1,1],K[0,2],K[1,2]
near = 0.01
far = 5.0

scaling_factor = 0.25
max_depth = far
h,w,fx,fy,cx,cy = jax3dp3.camera.scale_camera_parameters(orig_h,orig_w,orig_fx,orig_fy,orig_cx,orig_cy, scaling_factor)
depth = cv2.resize(depth, (w,h),interpolation=0)



jax3dp3.setup_renderer(h, w, fx, fy, cx, cy, near, far)

dirname = "ycb_downloaded_models"
model_dir = os.path.join(jax3dp3.utils.get_assets_dir(), dirname)
model_names = np.array(os.listdir(model_dir))
model_box_dims = []
for model in model_names:
    mesh = trimesh.load(os.path.join(jax3dp3.utils.get_assets_dir(), f"{dirname}/{model}/textured_simple.obj"))
    mesh = jax3dp3.mesh.center_mesh(mesh)
    model_box_dims.append(jax3dp3.utils.axis_aligned_bounding_box(mesh.vertices)[0])
    jax3dp3.load_model(mesh)
model_box_dims = jnp.array(model_box_dims)

table_face_param = 2
cam_pose = jnp.eye(4)

### segment the image
gt_image_full = t3d.depth_to_point_cloud_image(depth, fx,fy,cx,cy)
jax3dp3.viz.save_depth_image(gt_image_full[:,:,2], "gt_image_full.png", max=far)
gt_point_cloud_full = t3d.point_cloud_image_to_points(gt_image_full)
table_pose, table_dims = jax3dp3.utils.find_table_pose_and_dims(gt_point_cloud_full[gt_point_cloud_full[:,2] < far, :], ransac_threshold=0.001, inlier_threshold=0.002, segmentation_threshold=0.004)

gt_image_above_table = gt_image_full * (t3d.apply_transform(gt_image_full, jnp.linalg.inv(table_pose))[:,:,2] > 0.02)[:,:,None]
jax3dp3.viz.save_depth_image(gt_image_above_table[:,:,2], "gt_image_above_table.png", max=far)


segmentation_img = jax3dp3.utils.segment_point_cloud_image(gt_image_above_table, threshold=0.01, min_points_in_cluster=30)
jax3dp3.viz.save_depth_image(segmentation_img + 1, "seg.png", max=segmentation_img.max() + 1)
unique, counts =  np.unique(segmentation_img, return_counts=True)


### setup pose estimation
segmentation_idx_to_do_pose_estimation_for = unique[unique != -1]
print(segmentation_idx_to_do_pose_estimation_for)

poses_from_contact_params_sweep = jax.jit(jax.vmap(jax3dp3.scene_graph.pose_from_contact, in_axes=(0, None, 0, None, None, None)))
scorer_parallel_jit = jax.jit(jax.vmap(jax3dp3.likelihood.threedp3_likelihood, in_axes=(None, 0, None, None, None)))
    
object_indices = list(range(len(model_names)))
cf_pose_to_table_pose = lambda pose, table_plane_pose: t3d.inverse(table_plane_pose).dot(cam_pose) @ pose  # table_to_world * cam_to_world * pose_cf


def coarse_to_fine_contact_params(contact_param_sched, face_param_sched, likelihood_r_sched, init_latent_pose_table_frame, gt_image_masked, gt_img_complement, top_k=5):
    """
    do coarse-to-fine, keeping the top top_k hypotheses at each round
    """
    num_steps = len(likelihood_r_sched)
    latent_pose_cam_frame = t3d.inverse(t3d.inverse(table_pose).dot(cam_pose)) @ init_latent_pose_table_frame   
    top_k_objs = collections.deque([(float('-inf'), obj_idx, latent_pose_cam_frame) for obj_idx in object_indices])  # start with all objects inference
    
    for sched_i in range(num_steps):
        r = likelihood_r_sched[sched_i]

        contact_param_sweep_delta, face_param_sweep = contact_param_sched[sched_i], face_param_sched[sched_i]
                
        for _ in range(len(top_k_objs)):
            _, obj_idx, latent_pose_cam_frame = top_k_objs.popleft()

            # get contact params
            latent_pose_table_frame = t3d.inverse(table_pose).dot(cam_pose) @ latent_pose_cam_frame
            c = latent_pose_table_frame[:3, 3]
            contact_param_sweep = contact_param_sweep_delta + c  # shift center 

            # get pose proposals in cam frame
            pose_proposals = poses_from_contact_params_sweep(contact_param_sweep, table_face_param, face_param_sweep, table_dims, model_box_dims[obj_idx], table_pose)  

            # get best pose proposal
            images_unmasked = jax3dp3.render_parallel(pose_proposals, obj_idx)  # TODO multiobject parallelize
            images = jax3dp3.renderer.get_complement_masked_images(images_unmasked, gt_img_complement)
            weights = scorer_parallel_jit(gt_image_masked, images, r, 0.01, 20**3)
            best_pose_idx = weights.argmax()

            top_k_objs.append((weights[best_pose_idx], obj_idx, pose_proposals[best_pose_idx]))  

        if sched_i == 0: 
            top_k_objs = collections.deque(heapq.nlargest(top_k, top_k_objs))  # after the first iteration prune search down to the top_k top hypothesis objects 
        
        print(f"top {top_k} after {sched_i} iters:\n: {[(s, model_names[i]) for (s,i,p) in top_k_objs]}")

    best_score, best_obj_idx, best_pose_estimate_cam_frame = top_k_objs[0]
    
    return best_score, best_pose_estimate_cam_frame, best_obj_idx, top_k_objs
    


def get_pose_estimation_for_segmentation(seg_id):
    gt_image_masked = gt_image_above_table * (segmentation_img == seg_id)[:,:,None]
    gt_img_viz = jax3dp3.viz.get_depth_image(gt_image_masked[:,:,2],  max=max_depth)
    gt_img_viz.save("imgs/gt_image_masked.png")

    gt_img_complement = gt_image_above_table * (segmentation_img != seg_id)[:,:,None]
    gt_img_complement_viz = jax3dp3.viz.get_depth_image(gt_img_complement[:,:,2],  max=max_depth)
    gt_img_complement_viz.save("imgs/gt_img_complement.png")

    table_plane_pose = jax3dp3.scene_graph.get_contact_plane(table_pose, table_dims, table_face_param)
    points_in_table_ref_frame =  t3d.apply_transform(t3d.point_cloud_image_to_points(gt_image_masked), t3d.inverse(table_plane_pose).dot(cam_pose))
    point_seg = jax3dp3.utils.segment_point_cloud(points_in_table_ref_frame, 0.1)
    points_filtered = points_in_table_ref_frame[point_seg == jax3dp3.utils.get_largest_cluster_id_from_segmentation(point_seg)]
    center_x, center_y, _ = ( points_filtered.min(0) + points_filtered.max(0))/2
    
    top_k = 5
    # start = time.time() 

    grid_width = 0.2
    contact_param_sweep, face_param_sweep = jax3dp3.scene_graph.enumerate_contact_and_face_parameters(
        -grid_width, -grid_width, 0.0, +grid_width, +grid_width, jnp.pi, 
        11, 11, 11,
        jnp.array([3])
    )

    grid_width = 0.05
    contact_param_sweep_2, face_param_sweep_2 = jax3dp3.scene_graph.enumerate_contact_and_face_parameters(
        -grid_width, -grid_width, 0.0, +grid_width, +grid_width, jnp.pi, 
        11, 11, 11,
        jnp.array([3])
    )

    grid_width = 0.02
    contact_param_sweep_3, face_param_sweep_3 = jax3dp3.scene_graph.enumerate_contact_and_face_parameters(
        -grid_width, -grid_width, 0.0, +grid_width, +grid_width, jnp.pi, 
        11, 11, 11,
        jnp.array([3])
    )
    
    contact_param_sched = [contact_param_sweep, contact_param_sweep_2, contact_param_sweep_3]
    face_param_sched = [face_param_sweep, face_param_sweep_2, face_param_sweep_3]

    start = time.time()
    best_score, best_pose_estimate, best_idx, top_k_heap = coarse_to_fine_contact_params(contact_param_sched, face_param_sched, likelihood_r_sched=[0.06,0.04,0.02],
                                                                                        init_latent_pose_table_frame=t3d.transform_from_pos(jnp.array([center_x, center_y, 0])), 
                                                                                        gt_image_masked=gt_image_masked, gt_img_complement=gt_img_complement,
                                                                                        top_k=top_k)  
    print(best_score)
    end= time.time()
    print ("Time elapsed:", end - start)
    print(f"Best predicted obj = {model_names[best_idx]}")


    ## Viz
    all_scores = jnp.array([item[0] for item in heapq.nsmallest(top_k, top_k_heap)])
    all_names = jnp.array([item[1] for item in heapq.nsmallest(top_k, top_k_heap)])
    print(model_names[best_idx])
    filename = f"imgs/scene_{scene_num}_seg_id_{seg_id}.png"
    pred_rendered_img = jax3dp3.render_single_object(best_pose_estimate, best_idx)

    r_overlap_check = 0.05
    overlap = jax3dp3.likelihood.threedp3_likelihood_get_counts(gt_image_masked, pred_rendered_img, r_overlap_check)

    pred = jax3dp3.viz.get_depth_image(
        pred_rendered_img[:,:,2], max=max_depth
    )
    overlay = jax3dp3.viz.overlay_image(jax3dp3.viz.resize_image(rgb_viz, h,w), pred,alpha=0.5)
    
    bottom_text_string = "Object Class : Score\n"
    for i in np.argsort(-all_scores):
        bottom_text_string += (
            "{} : {}\n".format(model_names[all_names[i]], all_scores[i])
        )
    bottom_text_string += "\n"

    jax3dp3.viz.multi_panel([gt_img_viz, pred, overlay], 
        labels=[
            "Ground Truth", 
            "Prediction\nScore: {:.2f} {:s}".format(all_scores[best_idx], model_names[best_idx]), 
            "Overlap:\n{}/{}, {}/{}".format(
                *overlap
            )
        ],
        bottom_text=bottom_text_string,
        top_border=50,
        middle_width=50,
    ).save(filename)

for seg_id in segmentation_idx_to_do_pose_estimation_for:
    get_pose_estimation_for_segmentation(seg_id)
    # from IPython import embed; embed()

from IPython import embed; embed()



# for i in range(1,16):
#     file = open("./panda_dataset/scene_{}.pkl".format(i),'rb')
#     all_data = pickle.load(file)
#     file.close()

#     t = -1
#     data = all_data[t]
#     print(data.keys())

#     rgb = data["rgb"]
#     rgb_viz = jax3dp3.viz.get_rgb_image(rgb, 255.0)
#     rgb_viz.save("rgb_{}.png".format(i))
