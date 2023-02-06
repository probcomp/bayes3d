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
from threednel.utils.gcs import load_mesh


top_level_dir = jax3dp3.utils.get_assets_dir()
model_names = ["cracker_box", "sugar_box"]

data = np.load("data.npz")
rgb_images = data["rgb_images"]
depth_images = data["depth_images"]
poses = data["poses"]
poses = jnp.array(poses)

cam_pose = jnp.array(data["camera_pose"])
poses = t3d.inverse_pose(cam_pose) @ poses

orig_h, orig_w = data["camera_params"][:2]
orig_h, orig_w = int(orig_h), int(orig_w)

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

# import matplotlib.pyplot as plt

# fig, ax = plt.subplots(
#     1, len(bop_obj_indices), figsize=(10 * len(bop_obj_indices) + 1, 10)
# )
# for ii in range(len(bop_obj_indices)):
#     ax[ii].imshow(
#         model.get_emb_vis(torch.from_numpy(data_descriptors[:, :, ii])).cpu().numpy()
#     )
# plt.savefig("data_descriptors.png")

# jax3dp3.viz.vstack_images([
#     jax3dp3.viz.get_depth_image(data_xyz[:,:,2], max=far),
#     jax3dp3.viz.get_depth_image(obs_point_cloud_images[t,:,:,2], max=far)
# ]
# ).save("data.png")

offsets_ = []
scales_ = []
for bop_obj_idx in test_img.bop_obj_indices:
    obj_path = os.path.join(
        data_directory,
        f'bop/ycbv/models/obj_{bop_obj_idx:06d}.ply',
    )
    mesh = load_mesh(obj_path)
    bounding_sphere = mesh.bounding_sphere.primitive
    offset, scale = bounding_sphere.center, bounding_sphere.radius
    offsets_.append(offset)
    scales_.append(scale)

offsets = jnp.vstack(
    [
        jnp.zeros((1,3)),
        jnp.array(offsets_)
    ]
)
scales = jnp.hstack(
    [
        1.0, jnp.array(scales_)
    ]
)


# jax3dp3.setup_visualizer()
# jax3dp3.show_cloud("1", t3d.point_cloud_image_to_points(model_point_cloud_image_transformed[0]))
# jax3dp3.show_cloud("2", t3d.point_cloud_image_to_points(obj_coords_xyz), color=np.array([1.0, 0.0, 0.0]))

from IPython import embed; embed()

infer_mlp = siren_jax_from_surfemb(model)
from threednel.surfemb import utils


deltas = jax3dp3.make_translation_grid_enumeration(
    -50.0, -50.0, -50.0,
    50.0, 50.0, 50.0,
    5,5,5
)
r, outlier_prob, outlier_volume = 0.01, 0.01, 1**3

variances = jnp.diag(jnp.array([600.0, 10.0, 0.001]))
keys = jax.random.split(jax.random.PRNGKey(3), deltas.shape[0])
drift_poses = jax.vmap(jax3dp3.distributions.gaussian_pose, in_axes=(0,None))(keys, variances)
initial_pose_estimates = poses.copy()
particles = jnp.tile(initial_pose_estimates[:, None, :,:], (1, deltas.shape[0], 1,1))

all_particles = []
viz_images = []

r, outlier_prob, outlier_volume = 0.01, 0.01, 1**3

for t in range(obs_point_cloud_images.shape[0]):
    print(t)
    obj_idx = 1
    particles = particles.at[1,:].set(jnp.einsum("...ij,...jk->...ik", drift_poses, particles[1,:]))

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
    data_descriptors = surfemb.get_data_descriptors(test_img, scale=1.5, target_shape=(h,w))
    data_xyz, _ = depth_to_coords_in_camera(
        test_img.depth, test_img.intrinsics, as_image_shape=True
    )
    data_mask = test_img.depth > 0
    log_normalizers = voting.get_log_normalizers(
        data_descriptors, np.array(test_img.bop_obj_indices)
    )

    images = jax3dp3.render_multiobject_parallel(particles, [0,1])
    point_cloud_image = images[:,:,:,:3]
    model_point_cloud_image = jax3dp3.render_multiobject_parallel(particles, [0,1], on_object=1)[:,:,:,:3]
    segmentation_image = images[:,:,:,3] - 1.0

    segmentation_image_integer = jnp.round(segmentation_image).astype(jnp.int32)
    offsets_array = offsets[segmentation_image_integer + 1]
    scales_array = scales[segmentation_image_integer + 1]
    model_point_cloud_image_transformed = (model_point_cloud_image - offsets_array[...]) / scales_array[...,None]

    model_descriptors = jax.vmap(infer_mlp, in_axes=(0,0))(
        model_point_cloud_image_transformed.reshape(-1,3),
        (segmentation_image_integer + 1) .reshape(-1,1)
    )[:,0,0,0,0,:].reshape(segmentation_image_integer.shape[0],h,w,12)

    # model_descriptors_visualizations = utils.get_model_descriptors_visualizations(
    #     np.array(model_descriptors[0]),
    #     segmentation_image_integer[0],
    #     [
    #         voting.all_model_descriptors[bop_obj_idx]
    #         for bop_obj_idx in test_img.bop_obj_indices
    #     ],
    #     model,
    # )
    # plt.imshow(model_descriptors_visualizations)
    # plt.savefig("descriptors_new.png")

    ###### @Stannis replace this with the new likelihood.
    weights = jax3dp3.threedp3_likelihood_parallel_jit(
        obs_point_cloud_images[t] / 1000.0, images / 1000.0, r, outlier_prob, outlier_volume
    )
    parent_idxs = jax.random.categorical(keys[0], weights, shape=weights.shape)
    particles = particles[:,parent_idxs]
    keys = jax.random.split(keys[0], weights.shape[0])

    particle_image = jax3dp3.render_point_cloud(particles[1, :, :3, -1], h, w, fx,fy, cx,cy, near, far, 0)
    particle_image_viz = jax3dp3.viz.resize_image( jax3dp3.viz.get_depth_image(particle_image[:,:,2] > 0.0,max=2.0),orig_h,orig_w)

    reconstruction = jax3dp3.viz.resize_image(jax3dp3.viz.get_depth_image(jax3dp3.render_multiobject(particles[:,0], [0,1])[:,:,2], max=far),orig_h,orig_w)
    reconstruction_single_object = jax3dp3.viz.resize_image(jax3dp3.viz.get_depth_image(jax3dp3.render_single_object(particles[obj_idx], obj_idx)[:,:,2], max=far),orig_h,orig_w)
    rgb =  jax3dp3.viz.get_rgb_image(rgb_images[t],255.0)
    depth_viz = jax3dp3.viz.resize_image(jax3dp3.viz.get_depth_image(obs_point_cloud_images[t,:,:,2],max=far),orig_h,orig_w)

    particle_overlay_image_viz = jax3dp3.viz.overlay_image(rgb, particle_image_viz)

    rgb.save(f"imgs/{t}_rgb.png")
    depth_viz.save(f"imgs/{t}_depth.png")
    particle_overlay_image_viz.save(f"imgs/{t}_particles.png")
    jax3dp3.viz.overlay_image(rgb, reconstruction_single_object).save(f"imgs/{t}_overlay.png")

    viz_image = jax3dp3.viz.vstack_images([
        rgb,
        particle_overlay_image_viz,
        reconstruction,
        jax3dp3.viz.overlay_image(rgb, reconstruction_single_object)
    ])
    viz_image.save("img.png")
    viz_images.append(
        viz_image
    )
    all_particles.append(particles)

jax3dp3.viz.make_gif(viz_images, "out.gif")


from IPython import embed; embed()








# state.infer_table_plane(point_cloud_image, observation.camera_pose)


