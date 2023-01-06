import numpy as np
import jax.numpy as jnp
import jax
import jax3dp3
import time
from PIL import Image
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2
import trimesh
import os

h, w, fx,fy, cx,cy = (
    300,
    300,
    200.0,200.0,
    150.0,150.0
)
near,far = 0.001, 50.0
r = 0.1
outlier_prob = 0.01

jax3dp3.setup_renderer(h, w, fx, fy, cx, cy, near, far)

mesh = trimesh.load(os.path.join(jax3dp3.utils.get_assets_dir(),"cube.obj"))

jax3dp3.load_model(mesh)

num_frames = 50

gt_poses = [
    jnp.array([
    [1.0, 0.0, 0.0, -3.0],   
    [0.0, 1.0, 0.0, -3.0],   
    [0.0, 0.0, 1.0, 4.0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)
]
rot = R.from_euler('zyx', [1.0, -0.1, -2.0], degrees=True).as_matrix()
delta_pose =     jnp.array([
    [1.0, 0.0, 0.0, 0.09],   
    [0.0, 1.0, 0.0, 0.05],   
    [0.0, 0.0, 1.0, 0.02],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)
delta_pose = delta_pose.at[:3,:3].set(jnp.array(rot))

for t in range(num_frames):
    gt_poses.append(gt_poses[-1].dot(delta_pose))
gt_poses = jnp.stack(gt_poses)
print("gt_poses.shape", gt_poses.shape)



gt_images = jax3dp3.render_parallel(gt_poses,0)
print("gt_images.shape", gt_images.shape)
print((gt_images[0,:,:,-1] > 0 ).sum())

def scorer(rendered_image, gt, r, outlier_prob):
    weight = jax3dp3.likelihood.threedp3_likelihood(gt, rendered_image, r, outlier_prob)
    return weight
scorer_parallel = jax.vmap(scorer, in_axes=(0, None, None, None))
scorer_parallel_jit = jax.jit(scorer_parallel)

translation_deltas = jax3dp3.make_translation_grid_enumeration(-0.2, -0.2, -0.2, 0.2, 0.2, 0.2, 5, 5, 5)
key = jax.random.PRNGKey(3)
rotation_deltas = jax.vmap(lambda key: jax3dp3.distributions.gaussian_vmf(key, 0.00001, 800.0))(jax.random.split(key, 100))

pose_estimate = gt_poses[0]

start = time.time()
pose_estimates_over_time = []
for gt_image in gt_images:
    proposals = jnp.einsum("ij,ajk->aik", pose_estimate, translation_deltas)
    images = jax3dp3.render_parallel(proposals, 0)
    weights_new = scorer_parallel_jit(images, gt_image, 0.05, 0.1)
    pose_estimate = proposals[jnp.argmax(weights_new)]

    proposals = jnp.einsum("ij,ajk->aik", pose_estimate, rotation_deltas)
    images = jax3dp3.render_parallel(proposals, 0)
    weights_new = scorer_parallel_jit(images, gt_image, 0.05, 0.1)
    pose_estimate = proposals[jnp.argmax(weights_new)]

    pose_estimates_over_time.append(pose_estimate)
end = time.time()
print ("Time elapsed:", end - start)
print ("FPS:", gt_poses.shape[0] / (end - start))


viz_images = []
max_depth = 10.0
for i in range(gt_images.shape[0]):
    gt_img = jax3dp3.viz.get_depth_image(gt_images[i,:,:,2], max=max_depth)
    rendered_image = jax3dp3.render_single_object(pose_estimates_over_time[i], 0)
    rendered_img = jax3dp3.viz.get_depth_image(rendered_image[:,:,2], max=max_depth)
    viz_images.append(
        jax3dp3.viz.multi_panel(
            [gt_img, rendered_img],
            ["Ground Truth", "Inferred Reconstruction"],
            10,
            50,
            20
        )
    )
jax3dp3.viz.make_gif_from_pil_images(viz_images,"test.gif")

from IPython import embed; embed()