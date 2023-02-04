import matplotlib.pyplot as plt
import numpy as np
import os
from jax3dp3.viz import make_gif_from_pil_images
from PIL import Image
from copy import copy
from jax3dp3.viz import save_depth_image, get_depth_image, multi_panel
import jax3dp3.utils
import jax3dp3.viz
import jax3dp3.pybullet
import jax3dp3.transforms_3d as t3d
import jax.numpy as jnp

import sys
import cv2
import collections
import heapq
import sys

import warnings
import trimesh
import jax
import torch


top_level_dir = jax3dp3.utils.get_assets_dir()
model_names = ["cracker_box", "sugar_box"]

data = np.load("data.npz")
rgb_images = data["rgb_images"]
depth_images = data["depth_images"]
poses = data["poses"]
poses = jnp.array(poses)

cam_pose = t3d.transform_from_rot_and_pos(
    t3d.rotation_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), -jnp.pi/2),
    jnp.array([0.0, -500.0, 0.0])
)
poses = t3d.inverse_pose(cam_pose) @ poses

scale_factor = 0.3
state =jax3dp3.OnlineJax3DP3()
state.set_camera_parameters(data["camera_params"], scaling_factor=scale_factor)
(h,w,fx,fy,cx,cy,near,far) = state.camera_params

state.start_renderer()
for name in model_names:
    path = os.path.join(name, "textured.obj")
    state.add_trimesh(
        trimesh.load(path), mesh_name=name, mesh_scaling_factor=1.0
    )

obs_point_cloud_images = jnp.array([
    state.process_depth_to_point_cloud_image(d)
    for d in depth_images
])
obs_point_cloud_images = obs_point_cloud_images.at[obs_point_cloud_images[:,:,:,2] > far - 0.1,:].set(0.0)
jax3dp3.viz.save_depth_image(obs_point_cloud_images[0,:,:,2], "depth.png", max=far)


images = jax3dp3.render_multiobject(poses, [0,1])
jax3dp3.viz.save_depth_image(images[:,:,2], "depth2.png", max=far)

deltas = jax3dp3.make_translation_grid_enumeration(
    -50.0, -50.0, -50.0,
    50.0, 50.0, 50.0,
    5,5,5
)
r, outlier_prob, outlier_volume = 0.05, 0.01, 1**3


# jax3dp3.setup_visualizer()
# mesh = trimesh.load("/home/nishadgothoskar/data/bop/ycbv/models/obj_000003.ply")
# jax3dp3.show_cloud("c1", mesh.vertices / 1000.0 + 0.4, color=np.array([1.0, 0.0, 0.0]))

# jax3dp3.show_cloud("c2",state.meshes[1].vertices / 1000.0, color=np.array([0.0, 1.0, 0.0]))


import os

import joblib
import numpy as np
import taichi as ti
from threednel.bop.bop_surfemb import BOPSurfEmb
from threednel.bop.bop_vote import BOPVoting
from threednel.bop.data import RGBDImage
from threednel.likelihood import ndl
from threednel.likelihood.trace import Trace
from threednel.utils.transform import depth_to_coords_in_camera
from threednel.siren_jax import siren_jax_from_surfemb

# %%
data_directory = os.environ["BOP_DATA_DIR"]

ti.init(arch=ti.cuda)
surfemb = BOPSurfEmb(
    surfemb_model_path=os.path.join(
        data_directory, 'models', 'ycbv-jwpvdij1.compact.ckpt'
    ),
    device='cuda:0',
)
voting = BOPVoting(
    data_directory=data_directory,
)
model = surfemb.surfemb_model
voting.load_all_model_descriptors(model)


t = 0
rgb_img = rgb_images[t][:,:,:3]
depth_img = obs_point_cloud_images[t,:,:,2]
K = np.array([
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0],
])
bop_obj_indices = np.array([2,3])

test_img = RGBDImage(
    rgb_img, depth_img, K, bop_obj_indices
)
gl_renderer = test_img.get_renderer(data_directory)

data_descriptors = surfemb.get_data_descriptors(test_img, scale=1.5, target_shape=(h,w))
data_xyz, _ = depth_to_coords_in_camera(
    test_img.depth, test_img.intrinsics, as_image_shape=True
)
data_mask = test_img.depth > 0
log_normalizers = voting.get_log_normalizers(
    data_descriptors, np.array(test_img.bop_obj_indices)
)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(
    1, len(bop_obj_indices), figsize=(10 * len(bop_obj_indices) + 1, 10)
)
for ii in range(len(bop_obj_indices)):
    ax[ii].imshow(
        model.get_emb_vis(torch.from_numpy(data_descriptors[:, :, ii])).cpu().numpy()
    )
plt.savefig("data_descriptors.png")



jax3dp3.viz.vstack_images([
    jax3dp3.viz.get_depth_image(data_xyz[:,:,2], max=far),
    jax3dp3.viz.get_depth_image(obs_point_cloud_images[t,:,:,2], max=far)
]
).save("data.png")




initial_pose_estimates = poses.copy()
particles = jnp.tile(initial_pose_estimates[:, None, :,:], (1, deltas.shape[0], 1,1))
images = jax3dp3.render_multiobject_parallel(particles, [0,1])
point_cloud_image = images[:,:,:,:3]
model_point_cloud_image = jax3dp3.render_multiobject_parallel(particles, [0,1], on_object=1)[:,:,:,:3]
segmentation_image = images[:,:,:,3] - 1.0

gl_renderer = test_img.get_renderer(data_directory)

gt_poses = particles[:,0,:,:]
print(gt_poses.shape)
trace = Trace(ids=list(range(len(gt_poses))), poses=gt_poses)
model_xyz, obj_coords_xyz, model_descriptors, model_mask, obj_ids = ndl.get_model_info(
    trace, gl_renderer, model
)

jax3dp3.viz.vstack_images([
    jax3dp3.viz.get_depth_image(model_xyz[:,:,2], max=far),
    jax3dp3.viz.get_depth_image(images[t,:,:,2], max=far)
]
).save("rendered.png")

jax3dp3.viz.vstack_images([
    jax3dp3.viz.get_depth_image(obj_ids+1, max=4.0),
    jax3dp3.viz.get_depth_image(segmentation_image[t]+1, max=4.0)
]
).save("segids.png")

jax3dp3.viz.vstack_images([
    jax3dp3.viz.get_depth_image(obj_coords_xyz[:,:,2], max=far),
    jax3dp3.viz.get_depth_image(model_point_cloud_image[t,:,:,2], max=far)
]
).save("modelpoints.png")




offsets = jnp.vstack(
    [
        jnp.zeros((1,3)),
        jnp.array(gl_renderer.offsets)
    ]
)
scales = jnp.hstack(
    [
        1.0, jnp.array(gl_renderer.scales)
    ]
)
segmentation_image_integer = segmentation_image.astype(jnp.int32) + 1
offsets_array = offsets[segmentation_image_integer]
scales_array = scales[segmentation_image_integer]


model_point_cloud_image_transformed = (model_point_cloud_image - offsets_array[...]) / scales_array[...,None]

jax3dp3.viz.vstack_images([
    jax3dp3.viz.get_depth_image(obj_coords_xyz[:,:,2], max=3.0),
    jax3dp3.viz.get_depth_image(model_point_cloud_image_transformed[t,:,:,2], max=3.0)
]
).save("modelpoints.png")


jax3dp3.viz.get_depth_image(model_point_cloud_image_transformed[t,:,:,2] - obj_coords_xyz[:,:,2], max=0.1).save("errors.png")


from IPython import embed; embed()

jax3dp3.viz.vstack_images([
    jax3dp3.viz.get_depth_image(segmentation_image_integer[0], max=3.0),
    jax3dp3.viz.get_depth_image(obj_ids + 1, max=3.0)
]
).save("modelpoints.png")


infer_mlp = siren_jax_from_surfemb(model)
model_descriptors_me = jax.vmap(infer_mlp,in_axes=(0,0))(jnp.array([model_point_cloud_image_transformed[0]]).reshape(-1,3), jnp.array([segmentation_image_integer[0]]).reshape(-1,1))[:,0,0,0,0,:].reshape(1,h,w,12)
my_model_descriptors = model_descriptors_me[0]



from threednel.surfemb import utils
model_descriptors_visualizations = utils.get_model_descriptors_visualizations(
    np.array(my_model_descriptors),
    obj_ids,
    [
        voting.all_model_descriptors[bop_obj_idx]
        for bop_obj_idx in test_img.bop_obj_indices
    ],
    model,
)

fig, ax = plt.subplots(
    1,
    len(test_img.bop_obj_indices) + 1,
    figsize=(10 * len(test_img.bop_obj_indices) + 1, 10),
)
for ii in range(len(test_img.bop_obj_indices)):
    ax[ii].imshow(
        model.get_emb_vis(torch.from_numpy(data_descriptors[:, :, ii])).cpu().numpy()
    )

ax[-1].imshow(model_descriptors_visualizations)
fig.tight_layout()
fig.savefig("descriptors_old.png")





variances = jnp.diag(jnp.array([400.0, 100.0, 0.001]))
keys = jax.random.split(jax.random.PRNGKey(3), deltas.shape[0])
drift_poses = jax.vmap(jax3dp3.distributions.gaussian_pose, in_axes=(0,None))(keys, variances)

all_particles = []
viz_images = []
for t in range(obs_point_cloud_images.shape[0]):
    obj_idx = 1

    particles = particles.at[1,:].set(jnp.einsum("...ij,...jk->...ik", drift_poses, particles[1,:]))

    from IPython import embed; embed()
    images = jax3dp3.render_multiobject_parallel(particles, [0,1])
    point_cloud_image = images[:,:,:,:3]
    model_point_cloud_image = jax3dp3.render_multiobject_parallel(particles, [0,1], on_object=1)[:,:,:,:3]
    segmentation_image = images[:,:,:,3]

    data_descriptors = surfemb.get_data_descriptors(test_img)
    data_xyz, _ = depth_to_coords_in_camera(
        test_img.depth, test_img.intrinsics, as_image_shape=True
    )
    jax3dp3.viz.save_depth_image(segmentation_image[0], "segmentation.png", max=3.0)
    jax3dp3.viz.save_depth_image(segmentation_image[0], "segmentation.png", max=3.0)

    # jax3dp3.setup_visualizer()
    # jax3dp3.show_cloud("1", t3d.point_cloud_image_to_points(point_cloud_image[0])/1000.0)
    # jax3dp3.show_cloud("1", t3d.point_cloud_image_to_points(model_point_cloud_image[0])/1000.0)

    weights = jax3dp3.threedp3_likelihood_parallel_jit(
        obs_point_cloud_images[t] / 1000.0, images / 1000.0, r, outlier_prob, outlier_volume
    )
    parent_idxs = jax.random.categorical(keys[0], weights, shape=weights.shape)
    particles = particles[:,parent_idxs]
    keys = jax.random.split(keys[0], weights.shape[0])

    particle_image = jax3dp3.render_point_cloud(particles[1, :, :3, -1], h, w, fx,fy, cx,cy, near, far, 0)
    
    
    reconstruction = jax3dp3.viz.get_depth_image(obs_point_cloud_images[t][:,:,2],  max=far)
    rgb =  jax3dp3.viz.resize_image(jax3dp3.viz.get_rgb_image(rgb_images[t], 255.0),h,w)
    viz_images.append(
        jax3dp3.viz.multi_panel([
            rgb,
            jax3dp3.viz.get_depth_image(images[parent_idxs[0],:,:,2],max=far),
            jax3dp3.viz.get_depth_image(particle_image[:,:,2] > 0.0,max=2.0),
            reconstruction,
            jax3dp3.viz.overlay_image(rgb, reconstruction),
        ],
        labels=[
            "RGB",
            "Depth",
            "Particles",
            "Reconstruction",
            "Overlay",
        ])
    )
    all_particles.append(particles)

jax3dp3.viz.make_gif(viz_images, "out.gif")


from IPython import embed; embed()








# state.infer_table_plane(point_cloud_image, observation.camera_pose)


