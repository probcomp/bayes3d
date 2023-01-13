import numpy as np
import cv2
import jax3dp3.transforms_3d as t3d
import os
import jax
import jax.numpy as jnp
import trimesh
import time
import jax3dp3
import pickle
# jax3dp3.meshcat.setup_visualizer()

data = np.load("data_occluded.npz")
# data = np.load("data_visible.npz")

rgb = data["rgb"][0]
rgb_viz = jax3dp3.viz.get_rgba_image(rgb, 255.0)

segmentation = data["segmentation"][0]
depth = data["depth"][0]
_,_,orig_fx, orig_fy, orig_cx, orig_cy,near,far = data["params"]
orig_h,orig_w = depth.shape


scaling_factor = 0.5
max_depth = far
h,w,fx,fy,cx,cy = jax3dp3.camera.scale_camera_parameters(orig_h,orig_w,orig_fx,orig_fy,orig_cx,orig_cy, scaling_factor)
depth = cv2.resize(depth, (w,h),interpolation=0)

rgb_viz_resized = jax3dp3.viz.resize_image(rgb_viz, h,w)

gt_image_full = t3d.depth_to_point_cloud_image(depth, fx,fy,cx,cy)
jax3dp3.viz.save_depth_image(gt_image_full[:,:,2], "gt_image_full.png", max=far)
gt_point_cloud_full = t3d.point_cloud_image_to_points(gt_image_full)


# jax3dp3.meshcat.setup_visualizer()
# jax3dp3.meshcat.show_cloud("cloud", gt_point_cloud_full[gt_point_cloud_full[:,2] < far, :] / 10.0)


table_pose, table_dims = jax3dp3.utils.find_table_pose_and_dims(gt_point_cloud_full[gt_point_cloud_full[:,2] < far, :],
    ransac_threshold=0.01, inlier_threshold=0.2, segmentation_threshold=0.5
)

gt_image_above_table = gt_image_full * (t3d.apply_transform(gt_image_full, jnp.linalg.inv(table_pose))[:,:,2] > 0.2)[:,:,None]
jax3dp3.viz.save_depth_image(gt_image_above_table[:,:,2], "gt_image_above_table.png", max=far)

segmentation_img = jax3dp3.utils.segment_point_cloud_image(gt_image_above_table, threshold=0.4, min_points_in_cluster=5)
jax3dp3.viz.save_depth_image(segmentation_img + 1, "seg.png", max=segmentation_img.max() + 1)


model_box_dims = jnp.array([
    np.array([4.0, 1.5, 1.5]),
    np.array([2.0, 1.5, 1.5]),
    np.array([2.0, 0.5, 0.5]),
])
meshes = [
    jax3dp3.mesh.make_cuboid_mesh(dims)
    for dims in model_box_dims
]


jax3dp3.setup_renderer(h, w, fx, fy, cx, cy, near, far)
for mesh in meshes:
    jax3dp3.load_model(mesh)


seg_id = 1
gt_image_masked = gt_image_above_table * (segmentation_img == seg_id)[:,:,None]
gt_img_viz = jax3dp3.viz.get_depth_image(gt_image_masked[:,:,2],  max=max_depth)
gt_img_viz.save("gt_image_masked.png")

gt_img_complement = gt_image_above_table * (segmentation_img != seg_id)[:,:,None]
gt_img_complement_viz = jax3dp3.viz.get_depth_image(gt_img_complement[:,:,2],  max=max_depth)
gt_img_complement_viz.save("gt_img_complement.png")

grid_width = 3.1
contact_param_sweep, face_param_sweep = jax3dp3.scene_graph.enumerate_contact_and_face_parameters(
    -grid_width, -grid_width, 0.0, +grid_width, +grid_width, jnp.pi, 
    5, 5, 5,
    jnp.array([3])
)

grid_width = 1.0
contact_param_sweep_2, face_param_sweep_2 = jax3dp3.scene_graph.enumerate_contact_and_face_parameters(
    -grid_width, -grid_width, 0.0, +grid_width, +grid_width, jnp.pi, 
    5, 5, 5,
    jnp.array([3])
)

grid_width = 0.5
contact_param_sweep_3, face_param_sweep_3 = jax3dp3.scene_graph.enumerate_contact_and_face_parameters(
    -grid_width, -grid_width, 0.0, +grid_width, +grid_width, jnp.pi, 
    5, 5, 5,
    jnp.array([3])
)

grid_width = 0.1
contact_param_sweep_4, face_param_sweep_4 = jax3dp3.scene_graph.enumerate_contact_and_face_parameters(
    -grid_width, -grid_width, 0.0, +grid_width, +grid_width, jnp.pi, 
    5, 5, 5,
    jnp.array([3])
)

r = 0.5
outlier_prob, outlier_volume = 0.05, 10*20.*10.0


contact_param_sched = [contact_param_sweep, contact_param_sweep_2, contact_param_sweep_3, contact_param_sweep_4]
face_param_sched = [face_param_sweep, face_param_sweep_2, face_param_sweep_3, face_param_sweep_4]
likelihood_r_sched = [0.5, 0.5, 0.5,0.5]


table_face_param = 2
cam_pose = jnp.eye(4)
table_surface_plane_pose = jax3dp3.scene_graph.get_contact_plane(table_pose, table_dims, table_face_param)
points_in_table_ref_frame =  t3d.apply_transform(t3d.point_cloud_image_to_points(gt_image_masked), t3d.inverse(table_surface_plane_pose).dot(cam_pose))
point_seg = jax3dp3.utils.segment_point_cloud(points_in_table_ref_frame, 0.2)
points_filtered = points_in_table_ref_frame[point_seg == jax3dp3.utils.get_largest_cluster_id_from_segmentation(point_seg)]
center_x, center_y, _ = ( points_filtered.min(0) + points_filtered.max(0))/2


results = jax3dp3.c2f.c2f_contact_parameters(
    jnp.array([center_x, center_y, 0.0]),
    contact_param_sched, face_param_sched, likelihood_r_sched=likelihood_r_sched,
    contact_plane_pose=table_surface_plane_pose,
    gt_image_masked=gt_image_masked, gt_img_complement=gt_img_complement,
    model_box_dims=model_box_dims,
    outlier_prob=outlier_prob,
    outlier_volume=outlier_volume,
)

overlays = []
labels = []
scores = []
imgs = []
r_for_posterior = 0.5
for i in range(len(results)):
    score_orig, obj_idx, _, _, pose = results[i]
    image_unmasked = jax3dp3.render_single_object(pose, obj_idx)
    image = jax3dp3.renderer.get_complement_masked_image(image_unmasked, gt_img_complement)
    imgs.append(image)
    score = jax3dp3.threedp3_likelihood_parallel_jit(gt_image_masked, image[None, ...], r, outlier_prob, outlier_volume)[0]

    overlays.append(
        jax3dp3.viz.overlay_image(jax3dp3.viz.resize_image(rgb_viz_resized,h,w), jax3dp3.viz.get_depth_image(image_unmasked[:,:,2],  max=max_depth))
    )
    scores.append(score)
    labels.append(
        "Obj {:d}\n Score Orig: {:.2f} \n Score: {:.2f}".format(obj_idx, score_orig, score)
    )

normalized_probabilites = jax3dp3.utils.normalize_log_scores(jnp.array(scores))

jax3dp3.viz.multi_panel(
    [rgb_viz_resized, gt_img_viz, *overlays],
    labels=["RGB", "Depth Segment", *labels],
    bottom_text="{}\n Normalized Probabilites: {}".format(jnp.array(scores), jnp.round(normalized_probabilites, decimals=4)),
    label_fontsize =15,
    title="Class Ambiguity"
).save("out.png")



# jax3dp3.meshcat.clear()
# jax3dp3.meshcat.show_cloud("cloud", gt_image_masked / 10.0)
# jax3dp3.meshcat.show_cloud("cloud2", 
#     imgs[0] / 10.0,
#     color = np.array([1.0, 0.0, 0.0])
# )
# jax3dp3.meshcat.show_cloud("cloud3", 
#     imgs[1] / 10.0,
#     color = np.array([0.0, 1.0, 0.0])
# )


from IPython import embed; embed()
