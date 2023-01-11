
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


file = open("./panda_dataset/scene_2.pkl",'rb')
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


cube_mesh = trimesh.creation.box((0.02, 0.02, 0.02))
cube_dims = jax3dp3.utils.axis_aligned_bounding_box(cube_mesh.vertices)[0]
cube_mesh_id = len(model_names)
jax3dp3.load_model(cube_mesh)


gt_image_full = t3d.depth_to_point_cloud_image(depth, fx,fy,cx,cy)
jax3dp3.viz.save_depth_image(gt_image_full[:,:,2], "gt_image_full.png", max=far)
gt_point_cloud_full = t3d.point_cloud_image_to_points(gt_image_full)
table_pose, table_dims = jax3dp3.utils.find_table_pose_and_dims(gt_point_cloud_full[gt_point_cloud_full[:,2] < far, :],
    ransac_threshold=0.001, inlier_threshold=0.002, segmentation_threshold=0.004
)

gt_image_table = gt_image_full * (
    jnp.abs(t3d.apply_transform(gt_image_full, jnp.linalg.inv(table_pose))[:,:,2]) < 0.02
)[:,:,None]
jax3dp3.viz.save_depth_image(gt_image_table[:,:,2], "table.png", max=far)


gt_image_table_points = t3d.point_cloud_image_to_points(gt_image_table)
gt_image_table_points_in_table_frame = t3d.apply_transform(gt_image_table_points, jnp.linalg.inv(table_pose))
import cv2
(cx,cy), (width,height), rotation_deg = cv2.minAreaRect(np.array(gt_image_table_points_in_table_frame[:,:2]))
pose_shift = t3d.transform_from_rot_and_pos(
    t3d.rotation_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), jnp.deg2rad(rotation_deg)),
    jnp.array([cx,cy, 0.0])
)
table_pose = table_pose.dot(pose_shift)
import matplotlib.pyplot as plt
plt.scatter(gt_image_table_points_in_table_frame[:,0],gt_image_table_points_in_table_frame[:,1])
plt.savefig("scatter.png")

table_dims =  jnp.array([width, height, 1e-10])


contact_param_sweep = jax3dp3.enumerations.make_translation_grid_enumeration_3d(
     -table_dims[0]/2.0, -table_dims[1]/2.0, 0.0, table_dims[0]/2.0, table_dims[1]/2.0, 0.0, 30, 30, 1
)
poses_from_contact_params_sweep = jax.jit(jax.vmap(jax3dp3.scene_graph.pose_from_contact, in_axes=(0, None, None, None, None, None)))

pose_proposals_jit = jax.jit(lambda box_dims: poses_from_contact_params_sweep(contact_param_sweep, 2, 3, table_dims, box_dims, table_pose))
poses = pose_proposals_jit(cube_dims)


all_images_overlayed = jax3dp3.render_multiobject(poses, [cube_mesh_id for _ in range(poses.shape[0])])
enumeration_viz = jax3dp3.viz.get_depth_image(all_images_overlayed[:,:,2], max=far)
jax3dp3.viz.overlay_image(jax3dp3.viz.resize_image(rgb_viz, h,w), enumeration_viz).save("enumeration.png")


images_unmasked = jax3dp3.render_parallel(poses, cube_mesh_id)
keep_gt = jnp.logical_or(
    images_unmasked[:,:,:,2] == 0.0,
    (
        (gt_image_full[None,:,:,2] != 0.0) *
        (images_unmasked[:,:,:,2] >= gt_image_full[None,:,:,2])
    )
)[:,:,:,None]

# images_unmasked = 10, gt_iamge =0 ==> blocked=False ==> 10
# images_unmasked = 10, gt_iamge = 11 ==> blocked=False ==> 10
# images_unmasked = 11, gt_iamge = 9 ==> blocked=True ==> 9
# images_unmasked = 5, gt_iamge = 0 ==> blocked=False ==> 9
# images_unmasked = 0, gt_iamge = 5 ==> blocked=False ==> 0


images_apply_occlusions = (
    images_unmasked[:,:,:,:3] * (1- keep_gt) + 
    gt_image_full * keep_gt
)
jax3dp3.viz.save_depth_image(images_apply_occlusions[0, :,:,2], "img.png", 
    min=0.1,
    max=1.2
)
scorer_parallel_jit = jax.jit(jax.vmap(jax3dp3.likelihood.threedp3_likelihood, in_axes=(None, 0, None, None, None)))

r = 0.0001
weights = scorer_parallel_jit(gt_image_full, images_apply_occlusions, r, 0.001, 20**3)

order = jnp.argsort(-weights)

k = 300
top_k = order[:k]

all_images_overlayed = jax3dp3.render_multiobject(poses[top_k], [cube_mesh_id for _ in range(k)])
enumeration_viz = jax3dp3.viz.get_depth_image(all_images_overlayed[:,:,2], max=far)
enumeration_viz.save("best.png")
jax3dp3.viz.overlay_image(jax3dp3.viz.resize_image(rgb_viz, h,w), enumeration_viz).save("overlay.png")

from IPython import embed; embed()




# images = jax3dp3.render_parallel(pose_proposals, idx)



#     scorer_parallel_jit = jax.jit(jax.vmap(jax3dp3.likelihood.threedp3_likelihood, in_axes=(None, 0, None, None, None)))

    
#     grid_width = 0.05
#     contact_param_sweep, face_param_sweep = jax3dp3.scene_graph.enumerate_contact_and_face_parameters(
#         center_x-grid_width, center_y-grid_width, 0.0, center_x+grid_width, center_y+grid_width, jnp.pi*2, 9, 9, 10,
#         jnp.arange(6)
#     )
#     get_pose_proposals_jit = jax.jit(lambda box_dims: poses_from_contact_params_sweep(contact_param_sweep, table_face_param, face_param_sweep, table_dims, box_dims, table_pose))

#     object_indices = list(range(len(model_names)))
#     start = time.time()
#     all_scores = []
#     poses = []
#     r = 0.02
#     for idx in object_indices:
#         pose_proposals = get_pose_proposals_jit(model_box_dims[idx])
#         images = jax3dp3.render_parallel(pose_proposals, idx)
#         weights = scorer_parallel_jit(gt_image_masked, images, r, 0.01, 2**3)
#         best_pose_idx = weights.argmax()
#         all_scores.append(weights[best_pose_idx])
#         poses.append(pose_proposals[best_pose_idx])
#     all_scores = jnp.array(all_scores)
#     best_idx = np.argmax(all_scores)
#     end= time.time()
#     print ("Time elapsed:", end - start)

#     print(model_names[best_idx])
#     filename = "imgs/seg_id_{}.png".format(seg_id)
#     pred_rendered_img = jax3dp3.render_single_object(poses[best_idx], best_idx)

#     r_overlap_check = 0.05
#     overlap = jax3dp3.likelihood.threedp3_likelihood_get_counts(gt_image_masked, pred_rendered_img, r_overlap_check)

#     pred = jax3dp3.viz.get_depth_image(
#         pred_rendered_img[:,:,2], max=max_depth
#     )
#     overlay = jax3dp3.viz.overlay_image(jax3dp3.viz.resize_image(rgb_viz, h,w), pred,alpha=0.5)
    
#     bottom_text_string = "Object Class : Score\n"
#     for i in np.argsort(-all_scores):
#         bottom_text_string += (
#             "{} : {}\n".format(model_names[i], all_scores[i])
#         )
#     bottom_text_string += "\n"

#     jax3dp3.viz.multi_panel([gt_img_viz, pred, overlay], 
#         labels=[
#             "Ground Truth", 
#             "Prediction\nScore: {:.2f} {:s}".format(all_scores[best_idx], model_names[best_idx]), 
#             "Overlap:\n{}/{}, {}/{}".format(
#                 *overlap
#             )
#         ],
#         bottom_text=bottom_text_string,
#         top_border=50,
#         middle_width=50,
#     ).save(filename)



# from IPython import embed; embed()



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
