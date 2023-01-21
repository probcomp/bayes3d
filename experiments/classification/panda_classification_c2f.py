import numpy as np
import os
import jax
import jax.numpy as jnp
import trimesh
import time
import pickle
import jax3dp3.transforms_3d as t3d
import jax3dp3



# panda_dataset_path = os.path.join(jax3dp3.utils.get_assets_dir(), "panda_dataset")
# filename = "panda_dataset_2/ycb_no_background.pkl"
filename = "panda_dataset_2/ycb.pkl"
# filename = "panda_dataset/scene_4.pkl"
print(f"Processing scene {filename}...")
file = open(os.path.join(jax3dp3.utils.get_assets_dir(), filename),'rb')
all_data = pickle.load(file)
file.close()

t = 3
data = all_data[t]


rgb_original = data["rgb"]
depth_original = data["depth"] / 1000.0
K = data["intrinsics"][0]
orig_h,orig_w = depth_original.shape
K = jnp.array([[606.92871094, 0.    , 415.18270874],
    [ 0.    , 606.51989746, 258.89492798],
    [ 0.    , 0.    , 1.    ]])
orig_fx, orig_fy, orig_cx, orig_cy = K[0,0],K[1,1],K[0,2],K[1,2]
near = 0.001
far = 5.0

orig_h,orig_w = depth_original.shape
h,w,fx,fy,cx,cy = jax3dp3.camera.scale_camera_parameters(orig_h,orig_w,orig_fx,orig_fy,orig_cx,orig_cy, scaling_factor=0.3)
depth = jax3dp3.utils.resize(depth_original, h, w)
depth[depth > far] = 0.0


gt_image_full = t3d.depth_to_point_cloud_image(depth, fx, fy, cx, cy)


gt_point_cloud_full = t3d.point_cloud_image_to_points(gt_image_full)
table_pose, table_dims = jax3dp3.utils.find_table_pose_and_dims(
    gt_point_cloud_full[gt_point_cloud_full[:,2] < far, :], 
    ransac_threshold=0.001, inlier_threshold=0.002, segmentation_threshold=0.004
)
cam_pose = jnp.eye(4)

table_face_param = 2
table_surface_plane_pose = jax3dp3.scene_graph.get_contact_plane(table_pose, table_dims, table_face_param)


gt_image_in_table_frame = t3d.apply_transform(gt_image_full, jnp.linalg.inv(table_pose))
# jax3dp3.meshcat.setup_visualizer()
# jax3dp3.meshcat.show_cloud("c1",gt_image_in_table_frame)

above_table_mask = (
    (gt_image_in_table_frame[:,:,2] > 0.02) 
    # *
    # (jnp.abs(gt_image_in_table_frame[:,:,0]) < table_dims[0]) *
    # (jnp.abs(gt_image_in_table_frame[:,:,1]) < table_dims[1])
)
gt_image_above_table = gt_image_full * above_table_mask[:,:,None]
segmentation_img = jax3dp3.utils.segment_point_cloud_image(gt_image_above_table, threshold=0.01, min_points_in_cluster=30)
unique =  np.unique(segmentation_img)



jax3dp3.viz.multi_panel(
    [
        jax3dp3.viz.resize_image(jax3dp3.viz.get_rgb_image(rgb_original, 255.0),h,w),
        jax3dp3.viz.get_depth_image(gt_image_full[:,:,2],  max=far),
        jax3dp3.viz.get_depth_image(gt_image_above_table[:,:,2],  max=far),
        jax3dp3.viz.get_depth_image(segmentation_img + 1, max=segmentation_img.max() + 1),
    ],
    labels=["RGB", "Depth", "Above Table", "Segmentation"],
    bottom_text="Intrinsics {:d} {:d} {:0.2f} {:0.2f} {:0.2f} {:0.2f} {:0.2f} {:0.2f}\n".format(h,w,fx,fy,cx,cy,near,far),
).save("imgs/dashboard.png")

model_dir = os.path.join(jax3dp3.utils.get_assets_dir(), "bop/ycbv/models")
model_names = jax3dp3.ycb_loader.MODEL_NAMES
model_paths = []
for idx in range(21):
    model_paths.append(os.path.join(model_dir,"obj_" + f"{str(idx+1).rjust(6, '0')}.ply"))

model_scaling_factor= 1.0/1000.0

jax3dp3.setup_renderer(h, w, fx, fy, cx, cy, near, far)

model_box_dims = []
for path in model_paths:
    mesh = trimesh.load(path)  # 000001 to 000021
    mesh.vertices = mesh.vertices * model_scaling_factor
    model_box_dims.append(jax3dp3.utils.axis_aligned_bounding_box(mesh.vertices)[0])
    jax3dp3.load_model(mesh)
model_box_dims = jnp.array(model_box_dims)



outlier_prob=0.1
outlier_volume=10**3

contact_param_sched, face_param_sched = jax3dp3.c2f.make_schedules(
    grid_widths=[0.1, 0.07, 0.04, 0.02], grid_params=[(5, 5, 15),(5, 5, 15),(5, 5, 15),(5, 5, 15)]
)
likelihood_r_sched = [0.2, 0.1, 0.05, 0.02]




for seg_id in np.unique(segmentation_img):
    if seg_id == -1:
        continue

    print(seg_id)

    gt_image_masked = gt_image_above_table * (segmentation_img == seg_id)[:,:,None]
    gt_img_complement = gt_image_above_table * (segmentation_img != seg_id)[:,:,None]

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

    panel_viz = jax3dp3.c2f.multi_panel_c2f_viz(
        results, likelihood_r_sched[-1], gt_img_complement, gt_image_masked, 
        rgb_original, h, w, far, 
        outlier_prob, outlier_volume, 
        model_names, title=f"Likelihoods: {likelihood_r_sched}, Outlier Params: {outlier_prob},{outlier_volume}"
    )

    panel_viz.save(f"imgs/id_{seg_id}.png")

from IPython import embed; embed()
