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

file = open("./panda_dataset/scene_1.pkl",'rb')
all_data = pickle.load(file)
file.close()

t = -1
data = all_data[t]
print(data.keys())

rgb = data["rgb"]
jax3dp3.viz.get_rgb_image(rgb, 255.0).save("rgb.png")
depth = data["depth"] / 1000.0
jax3dp3.viz.get_depth_image(depth, max=3000.0).save("depth.png")
K = data["intrinsics"][0]
orig_h,orig_w = depth.shape
orig_fx, orig_fy, orig_cx, orig_cy = K[0,0],K[1,1],K[0,2],K[1,2]
near = 0.01
far = 5.0

scaling_factor = 0.3
max_depth = far
h,w,fx,fy,cx,cy = jax3dp3.camera.scale_camera_parameters(orig_h,orig_w,orig_fx,orig_fy,orig_cx,orig_cy, scaling_factor)
depth = cv2.resize(depth, (w,h),interpolation=0)



jax3dp3.setup_renderer(h, w, fx, fy, cx, cy, near, far)

model_dir = os.path.join(jax3dp3.utils.get_assets_dir(),"models")
model_names = np.array(os.listdir(model_dir))
model_box_dims = []
for model in model_names:
    mesh = trimesh.load(os.path.join(jax3dp3.utils.get_assets_dir(),"models/{}/textured_simple.obj".format(model)))
    mesh = jax3dp3.mesh.center_mesh(mesh)
    model_box_dims.append(jax3dp3.utils.axis_aligned_bounding_box(mesh.vertices)[0])
    jax3dp3.load_model(mesh)
model_box_dims = jnp.array(model_box_dims)




gt_image_full = t3d.depth_to_point_cloud_image(depth, fx,fy,cx,cy)
jax3dp3.viz.save_depth_image(gt_image_full[:,:,2], "gt_image_full.png", max=far)
gt_point_cloud_full = t3d.point_cloud_image_to_points(gt_image_full)
table_pose, table_dims = jax3dp3.utils.find_table_pose_and_dims(gt_point_cloud_full[gt_point_cloud_full[:,2] < far, :], ransac_threshold=0.001, inlier_threshold=0.002, segmentation_threshold=0.004)

gt_image_above_table = gt_image_full * (t3d.apply_transform(gt_image_full, jnp.linalg.inv(table_pose))[:,:,2] > 0.02)[:,:,None]
jax3dp3.viz.save_depth_image(gt_image_above_table[:,:,2], "gt_image_above_table.png", max=1.0)


segmentation_img = jax3dp3.utils.segment_point_cloud_image(gt_image_above_table, threshold=0.02, min_points_in_cluster=30)
jax3dp3.viz.save_depth_image(segmentation_img + 1, "seg.png", max=segmentation_img.max() + 1)
unique, counts =  np.unique(segmentation_img, return_counts=True)


gt_image_masked = gt_image_above_table * (segmentation_img == 1)[:,:,None]
gt_img_viz = jax3dp3.viz.get_depth_image(gt_image_masked[:,:,2],  max=max_depth)
gt_img_viz.save("gt_image_masked.png")

table_face_param = 2
cam_pose = jnp.eye(4)
table_plane_pose = jax3dp3.scene_graph.get_contact_plane(table_pose, table_dims, table_face_param)
points_in_table_ref_frame =  t3d.apply_transform(t3d.point_cloud_image_to_points(gt_image_masked), t3d.inverse(table_plane_pose).dot(cam_pose))
point_seg = jax3dp3.utils.segment_point_cloud(points_in_table_ref_frame, 0.1)
points_filtered = points_in_table_ref_frame[point_seg == jax3dp3.utils.get_largest_cluster_id_from_segmentation(point_seg)]
center_x, center_y, _ = ( points_filtered.min(0) + points_filtered.max(0))/2


grid_width = 0.05
contact_params_sweep = jax3dp3.make_translation_grid_enumeration_3d(center_x-grid_width, center_y-grid_width, 0.0, center_x+grid_width, center_y+grid_width, jnp.pi*2, 9, 9, 10)
contact_params_sweep_extended = jnp.tile(contact_params_sweep, (6,1))
face_params = jnp.hstack(
    [
        jnp.full((contact_params_sweep.shape[0]*6,),table_face_param).reshape(-1,1),
        jnp.repeat(jnp.arange(6),contact_params_sweep.shape[0]).reshape(-1,1)
    ]
)

poses_from_contact_params_sweep = jax.jit(jax.vmap(jax3dp3.scene_graph.pose_from_contact, in_axes=(0, 0, None, None, None)))
scorer_parallel_jit = jax.jit(jax.vmap(jax3dp3.likelihood.threedp3_likelihood, in_axes=(None, 0, None, None, None)))



object_indices = list(range(len(model_names)))
start = time.time()
all_scores = []
for idx in object_indices:
    pose_proposals = poses_from_contact_params_sweep(contact_params_sweep_extended, face_params, table_dims, model_box_dims[idx], table_pose)
    proposals = pose_proposals
    images = jax3dp3.render_parallel(proposals, idx)
    weights = scorer_parallel_jit(gt_image_masked, jnp.tile(images, (1,1,1,1)), 0.05, 0.1, 1**3)
    best_pose_idx = weights.argmax()
    filename = "imgs/best_{}.png".format(model_names[idx])
    pred = jax3dp3.viz.get_depth_image(
        images[best_pose_idx,:,:,2], max=max_depth
    )
    # pred.save(filename)
    jax3dp3.viz.overlay_image(gt_img_viz, pred,alpha=0.5).save(filename)
    all_scores.append(weights[best_pose_idx])
print(model_names[np.argmax(all_scores)])
end= time.time()
print ("Time elapsed:", end - start)

print(np.array(model_names)[np.argsort(all_scores)])
print(np.array(all_scores)[np.argsort(all_scores)])


from IPython import embed; embed()