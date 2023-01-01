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


data = np.load("../pybullet_world/data.npz")

h,w,fx,fy,cx,cy,near,far = data["params"]
h = int(h)
w = int(w)
max_depth = 20.0

segmentation = data["segmentation"]
depth = data["depth"]

scaling_factor = 0.25

h,w,fx,fy,cx,cy = jax3dp3.camera.scale_camera_paraeters(h,w,fx,fy,cx,cy, scaling_factor)


gt_image = depth_to_point_cloud_image(cv2.resize(depth * (segmentation == 1), (w,h),interpolation=0), fx,fy,cx,cy)
jax3dp3.viz.save_depth_image(gt_image[:,:,2], "gt_image.png", max=max_depth)

pgl.setup(h, w, fx, fy, cx, cy, near, far)

model_dir = os.path.join(jax3dp3.utils.get_assets_dir(),"models")
model_names = os.listdir(model_dir)
for model in model_names:
    model_path = os.path.join(jax3dp3.utils.get_assets_dir(),"models/{}/textured_simple.obj".format(model))
    pgl.load_model(model_path, h, w)

# r = 0.05
# outlier_prob = 0.1
def scorer(rendered_image, gt, r, outlier_prob):
    weight = jax3dp3.likelihood.threedp3_likelihood(gt, rendered_image, r, outlier_prob)
    return weight
scorer_parallel = jax.vmap(scorer, in_axes=(0, None, None, None))
scorer_parallel_jit = jax.jit(scorer_parallel)

import jax3dp3.bbox
non_zero_points = gt_image[gt_image[:,:,2]>0,:3]
_, initial_pose_estimate = jax3dp3.bbox.axis_aligned_bounding_box(non_zero_points)

translation_deltas = jax3dp3.enumerations.make_translation_grid_enumeration(-0.3, -0.3, -0.3, 0.3, 0.3, 0.3, 10, 10, 10)
translation_deltas_2 = jax3dp3.enumerations.make_translation_grid_enumeration(-0.1, -0.1, -0.1, 0.1, 0.1, 0.1, 10, 10, 10)
rotation_deltas = jax3dp3.enumerations.make_rotation_grid_enumeration(50, 20)


object_indices = list(range(len(model_names)))

start= time.time()
all_scores = []
for idx in object_indices:
    x = initial_pose_estimate.copy()
    proposals = jnp.einsum("ij,ajk->aik", x, translation_deltas)
    images = pgl.render(proposals, h,w,idx)
    weights = scorer_parallel_jit(images, gt_image, 0.1, 0.1)
    x = proposals[jnp.argmax(weights)]

    proposals = jnp.einsum("ij,ajk->aik", x, rotation_deltas)
    images = pgl.render(proposals, h,w,idx)
    weights = scorer_parallel_jit(images, gt_image, 0.05, 0.1)
    x = proposals[jnp.argmax(weights)]

    x = initial_pose_estimate.copy()
    proposals = jnp.einsum("ij,ajk->aik", x, translation_deltas_2)
    images = pgl.render(proposals, h,w,idx)
    weights = scorer_parallel_jit(images, gt_image, 0.05, 0.1)
    x = proposals[jnp.argmax(weights)]

    proposals = jnp.einsum("ij,ajk->aik", x, rotation_deltas)
    images = pgl.render(proposals, h,w,idx)
    weights = scorer_parallel_jit(images, gt_image, 0.1, 0.1)
    x = proposals[jnp.argmax(weights)]

    jax3dp3.viz.save_depth_image(images[weights.argmax(),:,:,2], "best_{}.png".format(model), max=max_depth)

    all_scores.append(weights.max())
print(model_names[np.argmax(all_scores)])
end= time.time()
print ("Time elapsed:", end - start)

print(np.array(model_names)[np.argsort(np.array(all_scores))])


from IPython import embed; embed()