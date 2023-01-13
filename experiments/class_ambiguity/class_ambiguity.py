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
jax3dp3.meshcat.setup_visualizer()

data = np.load("data.npz")

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




table_face_param = 2
cam_pose = jnp.eye(4)
table_plane_pose = jax3dp3.scene_graph.get_contact_plane(table_pose, table_dims, table_face_param)
points_in_table_ref_frame =  t3d.apply_transform(t3d.point_cloud_image_to_points(gt_image_masked), t3d.inverse(table_plane_pose).dot(cam_pose))
point_seg = jax3dp3.utils.segment_point_cloud(points_in_table_ref_frame, 0.2)
points_filtered = points_in_table_ref_frame[point_seg == jax3dp3.utils.get_largest_cluster_id_from_segmentation(point_seg)]
center_x, center_y, _ = ( points_filtered.min(0) + points_filtered.max(0))/2


poses_from_contact_params_sweep = jax.jit(jax.vmap(jax3dp3.scene_graph.pose_from_contact, in_axes=(0, None, 0, None, None, None)))
scorer_parallel_jit = jax.jit(jax.vmap(jax3dp3.likelihood.threedp3_likelihood, in_axes=(None, 0, None, None, None)))


grid_width = 3.1
contact_param_sweep, face_param_sweep = jax3dp3.scene_graph.enumerate_contact_and_face_parameters(
    -grid_width, -grid_width, 0.0, +grid_width, +grid_width, jnp.pi, 
    11, 11, 11,
    jnp.array([3])
)

grid_width = 1.0
contact_param_sweep_2, face_param_sweep_2 = jax3dp3.scene_graph.enumerate_contact_and_face_parameters(
    -grid_width, -grid_width, 0.0, +grid_width, +grid_width, jnp.pi, 
    11, 11, 11,
    jnp.array([3])
)



grid_width = 0.5
contact_param_sweep_3, face_param_sweep_3 = jax3dp3.scene_graph.enumerate_contact_and_face_parameters(
    -grid_width, -grid_width, 0.0, +grid_width, +grid_width, jnp.pi, 
    11, 11, 11,
    jnp.array([3])
)

r = 0.4
outlier_prob, outlier_volume = 0.1, 10**3

best_poses = []
idxs = []
all_scores = []
resolution = 0.001
def voxelize(data, resolution):
    return jnp.round(data /resolution) * resolution

for idx in [0,1, 2]:
    c = jnp.array([center_x, center_y, 0.0])
    for c_delta, f in [(contact_param_sweep, face_param_sweep),(contact_param_sweep_2, face_param_sweep_2), (contact_param_sweep_3, face_param_sweep_3)]:
        pose_proposals = poses_from_contact_params_sweep(c + c_delta, table_face_param, f, table_dims, 
                    model_box_dims[idx], table_pose)
        images_unmasked = jax3dp3.render_parallel(pose_proposals, idx)
        images = jax3dp3.renderer.get_complement_masked_images(images_unmasked, gt_img_complement)
        weights = scorer_parallel_jit(voxelize(gt_image_masked, resolution), voxelize(images, resolution), r, outlier_prob, outlier_volume)
        best_pose_idx = weights.argmax()
        c = (c+c_delta)[best_pose_idx]


    idxs.append(idx)
    best_poses.append(pose_proposals[best_pose_idx])


# scaling_factor = 0.5 
# max_depth = far
# h,w,fx,fy,cx,cy = jax3dp3.camera.scale_camera_parameters(h,w,fx,fy,cx,cy,scaling_factor)




# jax3dp3.setup_renderer(h, w, fx, fy, cx, cy, near, far)
# for mesh in meshes:
#     jax3dp3.load_model(mesh)


overlays = []
labels = []
scores = []
imgs = []
r = 2.0
resolution = 1.0
outlier_prob, outlier_volume = 0.2, 10**3
for i in range(len(idxs)):
    image_unmasked = jax3dp3.render_single_object(best_poses[i], idxs[i])
    image = jax3dp3.renderer.get_complement_masked_image(image_unmasked, gt_img_complement)
    imgs.append(image)
    score = scorer_parallel_jit(voxelize(gt_image_masked, resolution), voxelize(image, resolution)[None, ...], r, outlier_prob, outlier_volume)[0]

    overlays.append(
        jax3dp3.viz.overlay_image(jax3dp3.viz.resize_image(rgb_viz_resized,h,w), jax3dp3.viz.get_depth_image(image_unmasked[:,:,2],  max=max_depth))
    )
    scores.append(score)
    labels.append(
        "Obj {:d}\n Score: {:.2f}".format(idxs[i], score)
    )

normalized_probabilites = jax3dp3.utils.normalize_log_scores(jnp.array(scores))


jax3dp3.viz.multi_panel(
    [rgb_viz_resized, gt_img_viz, *overlays],
    labels=["RGB", "Depth Segment", *labels],
    bottom_text="{}\n Normalized Probabilites: {}".format(jnp.array(scores), jnp.round(normalized_probabilites, decimals=4)),
    label_fontsize =15,
    title="Class Ambiguity"
).save("out.png")



jax3dp3.meshcat.clear()
jax3dp3.meshcat.show_cloud("cloud", gt_image_masked / 10.0)
jax3dp3.meshcat.show_cloud("cloud2", 
    imgs[0] / 10.0,
    color = np.array([1.0, 0.0, 0.0])
)
jax3dp3.meshcat.show_cloud("cloud3", 
    imgs[1] / 10.0,
    color = np.array([0.0, 1.0, 0.0])
)


from IPython import embed; embed()
