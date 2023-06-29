import numpy as np
import jax.numpy as jnp
import jax
import bayes3d as j
import bayes3d as b
import time
from PIL import Image
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2
import trimesh
import os

# Can be helpful for debugging:
# jax.config.update('jax_enable_checks', True) 

intrinsics = b.Intrinsics(
    height=300,
    width=300,
    fx=200.0, fy=200.0,
    cx=150.0, cy=150.0,
    near=0.001, far=6.0
)

b.setup_renderer(intrinsics)
b.RENDERER.add_mesh_from_file(os.path.join(j.utils.get_assets_dir(),"sample_objs/cube.obj"))

num_frames = 60

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


gt_images = b.RENDERER.render_many(gt_poses[:,None,...],  jnp.array([0]))
print("gt_images.shape", gt_images.shape)
print("non-zero D-channel pixels in img 0:", (gt_images[0,:,:,-1] > 0 ).sum())

translation_deltas = j.make_translation_grid_enumeration(-0.2, -0.2, -0.2, 0.2, 0.2, 0.2, 5, 5, 5)
key = jax.random.PRNGKey(3)
rotation_deltas = jax.vmap(lambda key: j.distributions.gaussian_vmf(key, 0.00001, 800.0))(
    jax.random.split(key, 100)
)

# jax.vmap(b.RENDERER.render_parallel, in_axes=(0, None))

pose_estimate = gt_poses[0]


def update_pose_estimate(pose_estimate, gt_image):
    proposals = jnp.einsum("ij,ajk->aik", pose_estimate, translation_deltas)
    rendered_images = jax.vmap(b.RENDERER.render, in_axes=(0, None))(proposals[:,None, ...], jnp.array([0]))
    weights_new = b.threedp3_likelihood_parallel(gt_image, rendered_images, 0.05, 0.1, 10**3, 3)
    pose_estimate = proposals[jnp.argmax(weights_new)]

    proposals = jnp.einsum("ij,ajk->aik", pose_estimate, rotation_deltas)
    rendered_images = jax.vmap(b.RENDERER.render, in_axes=(0, None))(proposals[:, None, ...], jnp.array([0]))
    weights_new = b.threedp3_likelihood_parallel(gt_image, rendered_images, 0.05, 0.1, 10**3, 3)
    pose_estimate = proposals[jnp.argmax(weights_new)]
    return pose_estimate, pose_estimate

inference_program = jax.jit(lambda p,x: jax.lax.scan(update_pose_estimate, p,x)[1])
inferred_poses = inference_program(gt_poses[0], gt_images)

start = time.time()
pose_estimates_over_time = inference_program(gt_poses[0], gt_images)
end = time.time()
print ("Time elapsed:", end - start)
print ("FPS:", gt_poses.shape[0] / (end - start))


viz_images = []
max_depth = 10.0

rerendered_images = b.RENDERER.render_many(pose_estimates_over_time[:, None, ...], jnp.array([0]))
viz_images = [
    j.viz.multi_panel(
        [j.viz.get_depth_image(d[:,:,2]), j.viz.get_depth_image(r[:,:,2])],
        ["Ground Truth", "Inferred Reconstruction"],
    )
    for (r, d) in zip(rerendered_images, gt_images)
]
j.make_gif(viz_images, "demo.gif")

from IPython import embed; embed()