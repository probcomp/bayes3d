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

file = open("./panda_dataset/scene_6.pkl",'rb')
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
h,w = depth.shape
fx,fy,cx,cy = K[0,0],K[1,1],K[0,2],K[1,2]
near = 0.01
far = 5.0

scaling_factor = 0.5
max_depth = far
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
jax3dp3.viz.save_depth_image(gt_image_full[:,:,2], "gt_image_full.png", max=far)
gt_point_cloud_full = np.array(gt_image_full.reshape(-1,3))

plane_pose =  jax3dp3.utils.find_plane(
    gt_point_cloud_full[(0.0 < gt_point_cloud_full[:,2]) * (gt_point_cloud_full[:,2] < far),:],
    0.001
)
table_pose, table_dims = jax3dp3.utils.find_table_pose_and_dims(gt_point_cloud_full, plane_pose, 0.002, 0.004)
if table_pose[2,2] > 0:
    table_pose = table_pose @ t3d.transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.pi)

above_table_mask = t3d.apply_transform(gt_image_full, jnp.linalg.inv(table_pose))[:,:,2] > 0.0
gt_image_full = gt_image_full * above_table_mask[:,:,None]
jax3dp3.viz.save_depth_image(gt_image_full[:,:,2], "gt_image_full.png", max=far)
gt_point_cloud_full = np.array(gt_image_full.reshape(-1,3))

above_table_mask = t3d.apply_transform(gt_image_full, jnp.linalg.inv(table_pose))[:,:,2] > 0.02
non_table_img = gt_image_full * above_table_mask[:,:,None]
jax3dp3.viz.save_depth_image(non_table_img[:,:,2], "non_table_img.png", max=max_depth)


segmentation_img = jax3dp3.utils.segment_point_cloud_image(non_table_img, 0.04)
jax3dp3.viz.save_depth_image(segmentation_img + 1, "seg.png", max=segmentation_img.max()+1)
unique, counts =  np.unique(segmentation_img, return_counts=True)
print(unique[np.argsort(-counts)])

gt_image_single_object = gt_image_full * (segmentation_img ==2)[:,:,None]
gt_img_viz = jax3dp3.viz.get_depth_image(gt_image_single_object[:,:,2],  max=max_depth)
gt_img_viz.save("gt_image_single_object.png")


gt_image_single_object_cloud = t3d.point_cloud_image_to_points(gt_image_single_object)
cam_pose = jnp.eye(4)
gt_points_in_table_frame = t3d.apply_transform(
    t3d.apply_transform(gt_image_single_object_cloud, cam_pose), 
    jnp.linalg.inv(table_pose)
)

point_seg = jax3dp3.utils.segment_point_cloud(gt_points_in_table_frame, 10.0)
gt_points_in_table_frame_filtered = gt_points_in_table_frame[point_seg == 0]
import matplotlib.pyplot as plt
plt.clf()
plt.scatter(gt_points_in_table_frame_filtered[:,0],gt_points_in_table_frame_filtered[:,1])
plt.savefig("scatter.png")

center_x, center_y, _ = ( gt_points_in_table_frame_filtered.min(0) + gt_points_in_table_frame_filtered.max(0))/2

table_face_param = 2

grid_width = 0.1
contact_params_sweep = jax3dp3.make_translation_grid_enumeration_3d(center_x-grid_width, center_y-grid_width, 0.0, center_x+grid_width, center_y+grid_width, jnp.pi*2, 11, 11, 10)
poses_from_contact_params_sweep = jax.jit(jax.vmap(jax3dp3.scene_graph.pose_from_contact, in_axes=(0, None, None, None, None)))
scorer_parallel_jit = jax.jit(jax.vmap(jax3dp3.likelihood.threedp3_likelihood, in_axes=(None, 0, None, None, None)))



object_indices = list(range(len(model_names)))
start= time.time()
all_scores = []
model_indices = []
params = []
for idx in object_indices:
    for child_face in range(6):
        face_params = jnp.array([table_face_param, child_face])
        pose_proposals = poses_from_contact_params_sweep(contact_params_sweep, face_params, table_dims, model_box_dims[idx], table_pose)
        # proposals = jnp.einsum("ij,ajk->aik", jnp.linalg.inv(cam_pose), pose_proposals)
        proposals = pose_proposals
        images = jax3dp3.render_parallel(proposals, idx)
        weights = scorer_parallel_jit(gt_image_single_object, images, 0.05, 0.1, 1**3)
        best_pose_idx = weights.argmax()
        filename = "imgs/best_{}_face_{}.png".format(model_names[idx], child_face)
        pred = jax3dp3.viz.get_depth_image(
            images[best_pose_idx,:,:,2], max=max_depth
        )
        # pred.save(filename)
        jax3dp3.viz.overlay_image(gt_img_viz, pred,alpha=0.5).save(filename)
        all_scores.append(weights[best_pose_idx])
        model_indices.append(idx)
        params.append((child_face,))
print(model_names[model_indices[np.argmax(all_scores)]])
print(params[np.argmax(all_scores)])
end= time.time()
print ("Time elapsed:", end - start)

print(np.array(model_names)[np.argsort(all_scores)])
print(np.array(all_scores)[np.argsort(all_scores)])


from IPython import embed; embed()