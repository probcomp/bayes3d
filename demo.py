import numpy as np
import jax.numpy as jnp
import jax
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
    height=100,
    width=100,
    fx=50.0, fy=50.0,
    cx=50.0, cy=50.0,
    near=0.001, far=6.0
)

b.setup_renderer(intrinsics)
b.RENDERER.add_mesh_from_file(os.path.join(b.utils.get_assets_dir(),"sample_objs/bunny.obj"))

num_frames = 60

poses = [b.t3d.transform_from_pos(jnp.array([-3.0, 0.0, 3.5]))]
delta_pose = b.t3d.transform_from_rot_and_pos(
    R.from_euler('zyx', [-1.0, 0.1, 2.0], degrees=True).as_matrix(),
    jnp.array([0.09, 0.05, 0.02])
)
for t in range(num_frames-1):
    poses.append(poses[-1].dot(delta_pose))
poses = jnp.stack(poses)
print("Number of frames: ", poses.shape[0])

observed_images = b.RENDERER.render_many(poses[:,None,...],  jnp.array([0]))
print("observed_images.shape", observed_images.shape)

translation_deltas = b.utils.make_translation_grid_enumeration(-0.2, -0.2, -0.2, 0.2, 0.2, 0.2, 5, 5, 5)
rotation_deltas = jax.vmap(lambda key: b.distributions.gaussian_vmf_zero_mean(key, 0.00001, 800.0))(
    jax.random.split(jax.random.PRNGKey(30), 100)
)

likelihood = jax.vmap(b.threedp3_likelihood_old, in_axes=(None, 0, None, None, None, None, None))

def update_pose_estimate(pose_estimate, gt_image):
    proposals = jnp.einsum("ij,ajk->aik", pose_estimate, translation_deltas)
    rendered_images = jax.vmap(b.RENDERER.render, in_axes=(0, None))(proposals[:,None, ...], jnp.array([0]))
    weights_new = likelihood(gt_image, rendered_images, 0.05, 0.1, 10**3, 0.1, 3)
    pose_estimate = proposals[jnp.argmax(weights_new)]

    proposals = jnp.einsum("ij,ajk->aik", pose_estimate, rotation_deltas)
    rendered_images = jax.vmap(b.RENDERER.render, in_axes=(0, None))(proposals[:, None, ...], jnp.array([0]))
    weights_new = likelihood(gt_image, rendered_images, 0.05, 0.1, 10**3, 0.1, 3)
    pose_estimate = proposals[jnp.argmax(weights_new)]
    return pose_estimate, pose_estimate

inference_program = jax.jit(lambda p,x: jax.lax.scan(update_pose_estimate, p,x)[1])
inferred_poses = inference_program(poses[0], observed_images)

start = time.time()
pose_estimates_over_time = inference_program(poses[0], observed_images)
end = time.time()
print ("Time elapsed:", end - start)
print ("FPS:", poses.shape[0] / (end - start))

rerendered_images = b.RENDERER.render_many(pose_estimates_over_time[:, None, ...], jnp.array([0]))

viz_images = [
    b.viz.multi_panel(
        [
            b.viz.scale_image(b.viz.get_depth_image(d[:,:,2]), 3),
            b.viz.scale_image(b.viz.get_depth_image(r[:,:,2]), 3)
            ],
        labels=["Observed", "Rerendered"],
        label_fontsize=20
    )
    for (r, d) in zip(rerendered_images, observed_images)
]
b.make_gif_from_pil_images(viz_images, "assets/demo.gif")



from IPython import embed; embed()