import numpy as np
import jax.numpy as jnp
import jax
import bayes3d as j
import time
from PIL import Image
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2
import trimesh
import os

intrinsics = j.Intrinsics(
    height=300,
    width=300,
    fx=200.0, fy=200.0,
    cx=150.0, cy=150.0,
    near=0.001, far=6.0
)

renderer = j.Renderer(intrinsics)
renderer.add_mesh_from_file(os.path.join(j.utils.get_assets_dir(),"sample_objs/diamond.obj"))
renderer.add_mesh_from_file(os.path.join(j.utils.get_assets_dir(),"sample_objs/cube.obj"))

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



gt_images = jax.vmap(renderer.render_single_object, in_axes=(0, None))(gt_poses, jnp.int32(1))
gt_images = renderer.render_parallel(gt_poses, 1)
print("gt_images.shape", gt_images.shape)
print("non-zero D-channel pixels in img 0:", (gt_images[0,:,:,-1] > 0 ).sum())

translation_deltas = j.make_translation_grid_enumeration(-0.2, -0.2, -0.2, 0.2, 0.2, 0.2, 5, 5, 5)
key = jax.random.PRNGKey(3)
rotation_deltas = jax.vmap(lambda key: j.distributions.gaussian_vmf(key, 0.00001, 800.0))(
    jax.random.split(key, 100)
)

pose_estimate = gt_poses[0]

start = time.time()
pose_estimates_over_time = []
for gt_image in gt_images:
    proposals = jnp.einsum("ij,ajk->aik", pose_estimate, translation_deltas)
    rendered_images = renderer.render_parallel(proposals, 1)
    weights_new = j.threedp3_likelihood_parallel_jit(gt_image, rendered_images, 0.05, 0.1, 10**3, 3)
    pose_estimate = proposals[jnp.argmax(weights_new)]

    proposals = jnp.einsum("ij,ajk->aik", pose_estimate, rotation_deltas)
    rendered_images = renderer.render_parallel(proposals, 1)
    weights_new = j.threedp3_likelihood_parallel_jit(gt_image, rendered_images, 0.05, 0.1, 10**3, 3)
    pose_estimate = proposals[jnp.argmax(weights_new)]

    pose_estimates_over_time.append(pose_estimate)
end = time.time()
print ("Time elapsed:", end - start)
print ("FPS:", gt_poses.shape[0] / (end - start))


viz_images = []
max_depth = 10.0
for i in range(gt_images.shape[0]):
    gt_img = j.viz.get_depth_image(gt_images[i,:,:,2])
    rendered_image = renderer.render_single_object(pose_estimates_over_time[i], 1)
    rendered_img = j.viz.get_depth_image(rendered_image[:,:,2])
    viz_images.append(
        j.viz.multi_panel(
            [gt_img, rendered_img, j.overlay_image(gt_img, rendered_img)],
            ["Ground Truth", "Inferred Reconstruction", "Overlay"],
        )
    )
j.make_gif(viz_images, "demo.gif")


from IPython import embed; embed()