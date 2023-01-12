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

data = np.load("data.npz")

rgb = data["rgb"][0]
rgb_viz = jax3dp3.viz.get_rgba_image(rgb, 255.0)

segmentation = data["segmentation"][0]
depth = data["depth"][0]
_,_,orig_fx, orig_fy, orig_cx, orig_cy,near,far = data["params"]
orig_h,orig_w = depth.shape


scaling_factor = 0.25
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

segmentation_img = jax3dp3.utils.segment_point_cloud_image(gt_image_above_table, threshold=0.2, min_points_in_cluster=5)
jax3dp3.viz.save_depth_image(segmentation_img + 1, "seg.png", max=segmentation_img.max() + 1)


model_box_dims = jnp.array([
    np.array([4.0, 1.5, 1.5]),
    np.array([2.0, 1.5, 1.5])
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


grid_width = 3.0
contact_param_sweep, face_param_sweep = jax3dp3.scene_graph.enumerate_contact_and_face_parameters(
    center_x-grid_width, center_y-grid_width, jnp.pi/2, center_x+grid_width, center_y+grid_width, jnp.pi/2, 13, 13, 1,
    jnp.array([3])
)


object_indices = [0,1]

start = time.time()
all_scores = []
best_poses = []
r = 0.02
for idx in object_indices:
    pose_proposals = poses_from_contact_params_sweep(contact_param_sweep, table_face_param, face_param_sweep, table_dims, 
                    model_box_dims[idx], table_pose)
    print(pose_proposals)
    images_unmasked = jax3dp3.render_parallel(pose_proposals, idx)
    images = jax3dp3.renderer.get_complement_masked_images(images_unmasked, gt_img_complement)
    weights = scorer_parallel_jit(gt_image_masked, images, r, 0.01, 20**3)
    print(weights)
    best_pose_idx = weights.argmax()
    all_scores.append(weights[best_pose_idx])
    best_poses.append(pose_proposals[best_pose_idx])
all_scores = jnp.array(all_scores)
best_idx = np.argmax(all_scores)
print(best_idx)
end= time.time()
print ("Time elapsed:", end - start)

rerendered_img_viz = []
clouds = []
for idx in object_indices:
    print(all_scores[idx])
    best_img_unmasked = jax3dp3.render_single_object(best_poses[idx], idx)
    viz = jax3dp3.viz.get_depth_image(
        best_img_unmasked[:,:,2], max=max_depth
    )
    clouds.append(t3d.point_cloud_image_to_points(best_img_unmasked))
    overlay = jax3dp3.viz.overlay_image(rgb_viz_resized, viz)
    rerendered_img_viz.append(overlay)

jax3dp3.viz.multi_panel(
    [rgb_viz_resized, gt_img_viz, *rerendered_img_viz],
    fontsize=10
).save("out.png")


pose_proposals = poses_from_contact_params_sweep(contact_param_sweep, table_face_param, face_param_sweep, table_dims, 
                model_box_dims[idx], table_pose)
print(pose_proposals)
images_unmasked = jax3dp3.render_parallel(pose_proposals, idx)
images = jax3dp3.renderer.get_complement_masked_images(images_unmasked, gt_img_complement)


jax3dp3.meshcat.setup_visualizer()

jax3dp3.meshcat.clear()
jax3dp3.meshcat.show_cloud("cloud", clouds[0]/ 10.0)
jax3dp3.meshcat.show_cloud("cloud2", clouds[1]/ 10.0)
jax3dp3.meshcat.show_cloud("cloud3", gt_point_cloud_full / 10.0)


from IPython import embed; embed()


