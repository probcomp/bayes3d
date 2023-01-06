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
table_dims = jnp.array(data["table_dims"])
table_pose = jnp.array(data["table_pose"])
cam_pose = jnp.array(data["cam_pose"])[0]

h,w,fx,fy,cx,cy,near,far = data["params"]
h = int(h)
w = int(w)
max_depth = 20.0
scaling_factor = 0.25

h,w,fx,fy,cx,cy = jax3dp3.camera.scale_camera_parameters(h,w,fx,fy,cx,cy, scaling_factor)
jax3dp3.setup_renderer(h, w, fx, fy, cx, cy, near, far)

model_dir = os.path.join(jax3dp3.utils.get_assets_dir(),"models")
model_names = os.listdir(model_dir)
model_meshes = []
model_box_dims = []
for model in model_names:
    mesh = trimesh.load(os.path.join(jax3dp3.utils.get_assets_dir(),"models/{}/textured_simple.obj".format(model)))
    mesh = jax3dp3.mesh.center_mesh(mesh)
    model_box_dims.append(jax3dp3.utils.axis_aligned_bounding_box(mesh.vertices)[0])
    jax3dp3.load_model(mesh)


gt_image_full = t3d.depth_to_point_cloud_image(cv2.resize(depth, (w,h),interpolation=0), fx,fy,cx,cy)
jax3dp3.viz.save_depth_image(gt_image_full[:,:,2], "gt_image_full.png", max=max_depth)

gt_point_cloud = np.array(gt_image_full.reshape(-1,3))
plane_pose =  jax3dp3.utils.find_plane(gt_point_cloud[gt_point_cloud[:,2]<far,:], 0.02)
points_in_table_frame = t3d.apply_transform(gt_point_cloud, jnp.linalg.inv(plane_pose))
inliers = (jnp.abs(points_in_table_frame[:,2]) < 0.02)
inliers_img = inliers.reshape(gt_image_full.shape[:2])
jax3dp3.viz.save_depth_image(gt_image_full[:,:,2] * inliers_img, "gt_image_masked.png", max=max_depth)

from IPython import embed; embed()





gt_image = t3d.depth_to_point_cloud_image(cv2.resize(depth * (segmentation == 1), (w,h),interpolation=0), fx,fy,cx,cy)
jax3dp3.viz.save_depth_image(gt_image[:,:,2], "gt_image.png", max=max_depth)

contact_params = jnp.array([0.0, 0.0, -jnp.pi/4])
face_params = jnp.array([2,3])

contact_params_sweep = jax3dp3.make_translation_grid_enumeration_3d(-0.3, -0.3, 0.0, 0.3, 0.3, 2*jnp.pi, 11, 11, 8)

poses_from_contact_params_sweep = jax.jit(jax.vmap(jax3dp3.scene_graph.pose_from_contact, in_axes=(0, None, None, None, None)))

# r = 0.05
# outlier_prob = 0.1
def scorer(rendered_image, gt, r, outlier_prob):
    weight = jax3dp3.likelihood.threedp3_likelihood(gt, rendered_image, r, outlier_prob)
    return weight
scorer_parallel = jax.vmap(scorer, in_axes=(0, None, None, None))
scorer_parallel_jit = jax.jit(scorer_parallel)

object_indices = list(range(len(model_names)))
start= time.time()
all_scores = []
for idx in object_indices:
    pose_proposals = poses_from_contact_params_sweep(contact_params_sweep, face_params, table_dims, model_box_dims[idx], table_pose)
    proposals = jnp.einsum("ij,ajk->aik", jnp.linalg.inv(cam_pose), pose_proposals)
    images = jax3dp3.render_parallel(proposals, idx)
    weights = scorer_parallel_jit(images, gt_image, 0.05, 0.05)
    best_pose_idx = weights.argmax()
    jax3dp3.viz.save_depth_image(images[best_pose_idx,:,:,2], "imgs/best_{}.png".format(model_names[idx]), max=max_depth)
    all_scores.append(weights[best_pose_idx])
print(model_names[np.argmax(all_scores)])
end= time.time()
print ("Time elapsed:", end - start)

print(np.array(model_names)[np.argsort(all_scores)])
print(np.array(all_scores)[np.argsort(all_scores)])


from IPython import embed; embed()