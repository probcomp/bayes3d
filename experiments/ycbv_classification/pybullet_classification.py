import numpy as np
import cv2
from jax3dp3.transforms_3d import quaternion_to_rotation_matrix, depth_to_point_cloud_image
import jax3dp3.camera
import jax3dp3.viz
import jax3dp3.parallel_gl_renderer as pgl
import jax3dp3.utils
import os
import jax
import jax3dp3.likelihood
import jax3dp3.enumerations
import time
import jax.numpy as jnp
from jax3dp3.single_object_recognition import classify
import jax3dp3.scene_graph
import jax3dp3.bbox
import trimesh

data = np.load("data.npz")

h,w,fx,fy,cx,cy,near,far = data["params"]
h = int(h)
w = int(w)
max_depth = 20.0

segmentation = data["segmentation"]
depth = data["depth"]

scaling_factor = 0.25

h,w,fx,fy,cx,cy = jax3dp3.camera.scale_camera_paraeters(h,w,fx,fy,cx,cy, scaling_factor)
pgl.setup(h, w, fx, fy, cx, cy, near, far)

model_dir = os.path.join(jax3dp3.utils.get_assets_dir(),"models")
model_names = os.listdir(model_dir)
model_meshes = []
model_box_dims = []
for model in model_names:
    mesh = trimesh.load(os.path.join(jax3dp3.utils.get_assets_dir(),"models/{}/textured_simple.obj".format(model)))
    mesh = jax3dp3.mesh.center_mesh(mesh)
    model_box_dims.append(jax3dp3.bbox.axis_aligned_bounding_box(mesh.vertices)[0])
    pgl.load_model(mesh, h, w)

table_dims = jnp.array(data["table_dims"])
table_pose = jnp.array(data["table_pose"])
cam_pose = jnp.array(data["cam_pose"])


gt_image = depth_to_point_cloud_image(cv2.resize(depth * (segmentation == 2), (w,h),interpolation=0), fx,fy,cx,cy)
jax3dp3.viz.save_depth_image(gt_image[:,:,2], "gt_image.png", max=max_depth)

contact_params = jnp.array([0.0, 0.0, -jnp.pi/4])
face_params = jnp.array([2,3])

contact_params_sweep = jax3dp3.make_translation_grid_enumeration_3d(-0.3, -0.3, 0.0, 0.3, 0.3, 2*jnp.pi, 11, 11, 8)
# contact_params_sweep = jnp.tile(contact_params[None,...],(grid.shape[0],1))
# contact_params_sweep = contact_params_sweep.at[:,:2].set(grid)

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
    images = pgl.render(proposals, h,w, idx)
    weights = scorer_parallel_jit(images, gt_image, 0.05, 0.05)
    best_pose_idx = weights.argmax()
    jax3dp3.viz.save_depth_image(images[best_pose_idx,:,:,2], "best_{}.png".format(model_names[idx]), max=max_depth)
    all_scores.append(weights[best_pose_idx])
print(model_names[np.argmax(all_scores)])
end= time.time()
print ("Time elapsed:", end - start)

print(np.array(model_names)[np.argsort(all_scores)])
print(np.array(all_scores)[np.argsort(all_scores)])


from IPython import embed; embed()