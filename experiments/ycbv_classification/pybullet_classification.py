import numpy as np
import cv2
import jax3dp3.transforms_3d as t3d
import os
import jax
import jax.numpy as jnp
import trimesh
import time
import jax3dp3

data = np.load("data.npz")
segmentation = data["segmentation"][0]
depth = data["depth"][0]
# table_dims = jnp.array(data["table_dims"])
# table_pose = jnp.array(data["table_pose"])
# cam_pose = jnp.array(data["cam_pose"])[0]

h,w,fx,fy,cx,cy,near,far = data["params"]
h = int(h)
w = int(w)
depth[depth > far] = 0

scaling_factor = 0.5
max_depth = 10.0

h,w,fx,fy,cx,cy = jax3dp3.camera.scale_camera_parameters(h,w,fx,fy,cx,cy, scaling_factor)
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
jax3dp3.viz.save_depth_image(gt_image_full[:,:,2], "gt_image_full.png", max=5.0)
gt_point_cloud_full = np.array(gt_image_full.reshape(-1,3))

plane_pose =  jax3dp3.utils.find_plane(
    gt_point_cloud_full[(0.0 < gt_point_cloud_full[:,2]) * (gt_point_cloud_full[:,2] < far),:],
    0.001
)
table_pose, table_dims = jax3dp3.utils.find_table_pose_and_dims(gt_point_cloud_full, plane_pose, 0.02, 0.04)


points_in_plane_frame = t3d.apply_transform(gt_point_cloud_full, jnp.linalg.inv(table_pose))
inliers = (jnp.abs(points_in_plane_frame[:,2]) < 0.02)
inliers_img = inliers.reshape(gt_image_full.shape[:2])
non_table_img = gt_image_full * (1 -inliers_img)[:,:,None]
table_img = gt_image_full * (inliers_img)[:,:,None]
jax3dp3.viz.save_depth_image(non_table_img[:,:,2], "non_table_img.png", max=max_depth)
jax3dp3.viz.save_depth_image(table_img[:,:,2], "table_img.png", max=max_depth)


segmentation_img = jax3dp3.utils.segment_point_cloud_image(non_table_img, 0.04)
jax3dp3.viz.save_depth_image(segmentation_img + 1, "seg.png", max=5.0)


gt_image_single_object = gt_image_full * (segmentation_img == 1)[:,:,None]
gt_img_viz = jax3dp3.viz.get_depth_image(gt_image_single_object[:,:,2],  max=max_depth)
gt_img_viz.save("gt_image_single_object.png")

gt_image_single_object_cloud = jax3dp3.utils.point_cloud_image_to_points(gt_image_single_object)
center_x, center_y, _ = jnp.mean(t3d.apply_transform(gt_image_single_object_cloud, jnp.linalg.inv(table_pose)),axis=0)

table_face_param = 2

face_params = jnp.array([table_face_param,3])

grid_width = 0.1
contact_params_sweep = jax3dp3.make_translation_grid_enumeration_3d(center_x-grid_width, center_y-grid_width, 0.0, center_x+grid_width, center_y+grid_width, jnp.pi*2, 11, 11, 36)
poses_from_contact_params_sweep = jax.jit(jax.vmap(jax3dp3.scene_graph.pose_from_contact, in_axes=(0, None, None, None, None)))
scorer_parallel_jit = jax.jit(jax.vmap(jax3dp3.likelihood.threedp3_likelihood, in_axes=(0, None, None, None)))



object_indices = list(range(len(model_names)))
start= time.time()
all_scores = []
for idx in object_indices:
    pose_proposals = poses_from_contact_params_sweep(contact_params_sweep, face_params, table_dims, model_box_dims[idx], table_pose)
    # proposals = jnp.einsum("ij,ajk->aik", jnp.linalg.inv(cam_pose), pose_proposals)
    proposals = pose_proposals
    images = jax3dp3.render_parallel(proposals, idx)
    weights = scorer_parallel_jit(images, gt_image_single_object, 0.02, 0.1)
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