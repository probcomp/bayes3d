import numpy as np
import os
import jax
import jax.numpy as jnp
import trimesh
import time
import pickle
import jax3dp3.transforms_3d as t3d
import jax3dp3

# for scene_num in range(1,16):
for scene_num in [1,3,4,6]:
    print(f"Processing scene {scene_num}...")
    panda_dataset_path = os.path.join(jax3dp3.utils.get_assets_dir(), "panda_dataset")
    file = open(os.path.join(panda_dataset_path, f"scene_{scene_num}.pkl"),'rb')
    all_data = pickle.load(file)
    file.close()
    t = -1
    data = all_data[t]

    rgb = data["rgb"]
    rgb_viz = jax3dp3.viz.get_rgb_image(rgb, 255.0)
    rgb_viz.save("rgb.png")
    depth = data["depth"] / 1000.0
    jax3dp3.viz.get_depth_image(depth, max=3000.0).save("depth.png")
    K = data["intrinsics"][0]
    orig_h,orig_w = depth.shape
    orig_fx, orig_fy, orig_cx, orig_cy = K[0,0],K[1,1],K[0,2],K[1,2]
    near = 0.001
    far = 5.0

    scaling_factor = 0.3
    max_depth = far
    h,w,fx,fy,cx,cy = jax3dp3.camera.scale_camera_parameters(orig_h,orig_w,orig_fx,orig_fy,orig_cx,orig_cy, scaling_factor)
    depth = jax3dp3.utils.resize(depth, h, w)
    rgb_viz_resized = jax3dp3.viz.resize_image(rgb_viz, h,w)


    jax3dp3.setup_renderer(h, w, fx, fy, cx, cy, near, far)

    num_models = 21
    model_dir = os.path.join(jax3dp3.utils.get_assets_dir(), "bop/ycbv/models")
    model_names = jax3dp3.ycb_loader.MODEL_NAMES

    model_box_dims = []
    for idx in range(num_models):
        mesh = trimesh.load(os.path.join(model_dir,"obj_" + f"{str(idx+1).rjust(6, '0')}.ply"))  # 000001 to 000021
        mesh.vertices = mesh.vertices / 1000.0
        model_box_dims.append(jax3dp3.utils.axis_aligned_bounding_box(mesh.vertices)[0])
        jax3dp3.load_model(mesh)
    model_box_dims = jnp.array(model_box_dims)
    
    cam_pose = jnp.eye(4)


    ### segment the image
    gt_image_full = t3d.depth_to_point_cloud_image(depth, fx,fy,cx,cy)
    jax3dp3.viz.save_depth_image(gt_image_full[:,:,2], "gt_image_full.png", max=far)
    gt_point_cloud_full = t3d.point_cloud_image_to_points(gt_image_full)
    table_pose, table_dims = jax3dp3.utils.find_table_pose_and_dims(gt_point_cloud_full[gt_point_cloud_full[:,2] < far, :], 
    ransac_threshold=0.001, inlier_threshold=0.002, segmentation_threshold=0.004)

    gt_image_above_table = gt_image_full * (t3d.apply_transform(gt_image_full, jnp.linalg.inv(table_pose))[:,:,2] > 0.02)[:,:,None]
    jax3dp3.viz.save_depth_image(gt_image_above_table[:,:,2], "gt_image_above_table.png", max=far)

    segmentation_img = jax3dp3.utils.segment_point_cloud_image(gt_image_above_table, threshold=0.01, min_points_in_cluster=30)
    jax3dp3.viz.save_depth_image(segmentation_img + 1, "seg.png", max=segmentation_img.max() + 1)
    unique, counts =  np.unique(segmentation_img, return_counts=True)

    table_face_param = 2
    table_surface_plane_pose = jax3dp3.scene_graph.get_contact_plane(table_pose, table_dims, table_face_param)

    segmentation_idx_to_do_pose_estimation_for = unique[unique != -1]
    print(segmentation_idx_to_do_pose_estimation_for)

    object_indices = list(range(len(model_names)))


    grid_width = 0.1
    contact_param_sweep, face_param_sweep = jax3dp3.scene_graph.enumerate_contact_and_face_parameters(
        -grid_width, -grid_width, 0.0, +grid_width, +grid_width, jnp.pi*2, 
        5, 5, 15,
        jnp.arange(6)
    )

    grid_width = 0.07
    contact_param_sweep_2, face_param_sweep_2 = jax3dp3.scene_graph.enumerate_contact_and_face_parameters(
        -grid_width, -grid_width, 0.0, +grid_width, +grid_width, jnp.pi*2, 
        5, 5, 15,
        jnp.arange(6)
    )

    grid_width = 0.04
    contact_param_sweep_3, face_param_sweep_3 = jax3dp3.scene_graph.enumerate_contact_and_face_parameters(
        -grid_width, -grid_width, 0.0, +grid_width, +grid_width, jnp.pi*2, 
        5, 5, 15,
        jnp.arange(6)
    )

    grid_width = 0.02
    contact_param_sweep_4, face_param_sweep_4 = jax3dp3.scene_graph.enumerate_contact_and_face_parameters(
        -grid_width, -grid_width, 0.0, +grid_width, +grid_width, jnp.pi*2, 
        5, 5, 15,
        jnp.arange(6)
    )

    contact_param_sched = [contact_param_sweep, contact_param_sweep_2, contact_param_sweep_3, contact_param_sweep_4]
    face_param_sched = [face_param_sweep, face_param_sweep_2, face_param_sweep_3, face_param_sweep_4]
    likelihood_r_sched = [0.2, 0.1, 0.05, 0.02]

    outlier_prob, outlier_volume = 0.1, 10**3

    def get_pose_estimation_for_segmentation(seg_id):
        gt_image_masked = gt_image_above_table * (segmentation_img == seg_id)[:,:,None]
        gt_img_viz = jax3dp3.viz.get_depth_image(gt_image_masked[:,:,2],  max=max_depth)
        gt_img_viz.save("imgs/gt_image_masked.png")

        gt_img_complement = gt_image_above_table * (segmentation_img != seg_id)[:,:,None]
        gt_img_complement_viz = jax3dp3.viz.get_depth_image(gt_img_complement[:,:,2],  max=max_depth)
        gt_img_complement_viz.save("imgs/gt_img_complement.png")


        points_in_table_ref_frame =  t3d.apply_transform(t3d.point_cloud_image_to_points(gt_image_masked), 
            t3d.inverse(table_surface_plane_pose).dot(cam_pose))
        point_seg = jax3dp3.utils.segment_point_cloud(points_in_table_ref_frame, 0.1)
        points_filtered = points_in_table_ref_frame[point_seg == jax3dp3.utils.get_largest_cluster_id_from_segmentation(point_seg)]
        center_x, center_y, _ = ( points_filtered.min(0) + points_filtered.max(0))/2
        
        top_k = 5

        start = time.time()
        results = jax3dp3.c2f.c2f_contact_parameters(
            jnp.array([center_x, center_y, 0.0]),
            contact_param_sched, face_param_sched, likelihood_r_sched=likelihood_r_sched,
            contact_plane_pose=table_surface_plane_pose,
            gt_image_masked=gt_image_masked, gt_img_complement=gt_img_complement,
            model_box_dims=model_box_dims,
            outlier_prob=outlier_prob,
            outlier_volume=outlier_volume,
            top_k=top_k
        )
        end= time.time()
        print ("Time elapsed:", end - start)


        panel_viz = jax3dp3.c2f.multi_panel_c2f(
            results, likelihood_r_sched[-1], gt_img_complement, gt_image_masked, 
            rgb_viz, h, w, max_depth, 
            outlier_prob, outlier_volume, 
            model_names, title=f"Likelihoods: {likelihood_r_sched}, Outlier Params: {outlier_prob},{outlier_volume}"
        )

        panel_viz.save(f"imgs/scene_{scene_num}_id_{seg_id}.png")

    for seg_id in segmentation_idx_to_do_pose_estimation_for:
        get_pose_estimation_for_segmentation(seg_id)
        # from IPython import embed; embed()

from IPython import embed; embed()
