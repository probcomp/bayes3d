import argparse
import os
from functools import partial

import cv2
import joblib
import numpy as np
import pandas as pd
import taichi as ti
import torch.utils.data
from objects3d.bop.bop_surfemb import BOPSurfEmb
from objects3d.bop.bop_vote import BOPVoting
from objects3d.bop.data import BOPTestDataset
from objects3d.likelihood import ndl as jax_ndl
from objects3d.likelihood.taichi_ndl import TaichiNDL
from objects3d.likelihood.trace import Trace
from objects3d.likelihood.utils import get_model_info
from objects3d.inference.inference_program import inference_program
from objects3d.likelihood import ndl
from objects3d.likelihood.trace import Trace
from objects3d.surfemb import utils
from objects3d.surfemb.surface_embedding import SurfaceEmbeddingModel
from objects3d.utils.transform import depth_to_coords_in_camera
from objects3d.visualization.images import save_multiple_images, save_rgb_image, save_depth_image
from objects3d.surfemb.siren_jax import siren_jax_from_surfemb
from objects3d.bop.data import BOPTestImage

ti.init(arch=ti.vulkan)
device = torch.device('cuda:0')

data_directory = os.path.expanduser('~/data')
surfemb_model_path = os.path.join(data_directory, 'models/ycbv-jwpvdij1.compact.ckpt')
dataset = 'ycbv'

data = np.load("data.npz")
depth_imgs = np.array(data["depth_imgs"]).copy()
rgb_imgs = np.array(data["rgb_imgs"]).copy()
scaling_factor = 1.0

fx = data["fx"] * scaling_factor
fy = data["fy"] * scaling_factor

cx = data["cx"] * scaling_factor
cy = data["cy"] * scaling_factor

original_height = data["height"]
original_width = data["width"]

h = int(np.round(original_height  * scaling_factor))
w = int(np.round(original_width * scaling_factor))
print(h,w,fx,fy,cx,cy)

ti.init(arch=ti.cuda)
surfemb = BOPSurfEmb(data_directory=data_directory, dataset=dataset, device='cuda:0')

data = BOPTestDataset(
    data_directory=data_directory,
    dataset=dataset,
    load_detector_crops=True,
)
voting = BOPVoting(
    data_directory=data_directory,
    dataset=dataset,
)
model = surfemb.surfemb_model
voting.load_all_model_descriptors(model)
all_model_descriptors = voting.all_model_descriptors

intrinsics = np.array([
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0]
])
cam_pose = np.eye(4)


scene_id = 48
scale_factor = 0.25
test_scene = data[scene_id]
img_id = test_scene.img_indices[1]
test_img = test_scene[img_id]

bop_obj_indices = [2,3]
test_img = BOPTestImage(
    dataset='ycbv',
    scene_id=-1,
    img_id=-1,
    rgb=rgb_imgs[0][:,:,:3],
    depth=depth_imgs[0],
    intrinsics=intrinsics,
    camera_pose=cam_pose,
    bop_obj_indices=tuple(bop_obj_indices),
    default_scales=np.array(
        [test_scene.default_scales[bop_obj_idx] for bop_obj_idx in bop_obj_indices]
    ),
    annotations=None
)


rgb = test_img.rgb
depth = test_img.depth.copy()
depth[depth > voting.voting_config.max_depth] = 0
intrinsics = test_img.intrinsics.copy()
cam_pose = test_img.camera_pose.copy()
bop_obj_indices = test_img.bop_obj_indices
n_objs = len(bop_obj_indices)
gl_renderer = test_img.get_renderer(data_directory, 1.0)

gt_poses = [
        np.array([
            [1.0, 0.0, 0.0, 0.00],
            [0.0, 1.0, 0.0, -1.1],
            [0.0, 0.0, 1.0, 500.0],
            [0.0, 0.0, 0.0, 1.0],
        ]),
        np.array([
            [1.0, 0.0, 0.0, -4.00],
            [0.0, 1.0, 0.0, 0.00],
            [0.0, 0.0, 1.0, 505.0],
            [0.0, 0.0, 0.0, 1.0],
        ]),
    ]


gt_depth = gl_renderer.render(np.arange(n_objs), gt_poses)[0][..., 2]
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(40, 10))
ax[0].imshow(rgb)
ax[1].imshow(depth)
ax[2].imshow(gt_depth, cmap='turbo')
fig.tight_layout()
fig.savefig("data.png")

# # %%
# img_id

# # %%
# data_xyz, _ = depth_to_coords_in_camera(depth, intrinsics, as_image_shape=True)
# data_mask = depth > 0
# print(data_mask.sum())

# # %%
# data_descriptors, pose_hypotheses = get_hypotheses_from_detector_crops(
#     test_img,
#     surfemb,
#     voting,
#     n_pose_hypotheses_per_crop=n_pose_hypotheses_per_crop,
#     n_top_rotations_per_translation=2,
#     n_pose_hypotheses_per_object=30,
#     maximum_filter_size=10,
# )

# # %%
# trace = Trace(ids=list(range(len(gt_poses))), poses=gt_poses)
# _, model_descriptors, _, obj_ids = get_model_info(trace, gl_renderer, model)
# model_descriptors_visualizations = utils.get_model_descriptors_visualizations(
#     model_descriptors,
#     obj_ids,
#     [all_model_descriptors[bop_obj_idx] for bop_obj_idx in bop_obj_indices],
#     model,
# )
# fig, ax = plt.subplots(
#     1,
#     len(test_img.bop_obj_indices) + 1,
#     figsize=(10 * (len(test_img.bop_obj_indices) + 1), 15),
# )
# for ii in range(len(test_img.bop_obj_indices)):
#     ax[ii].imshow(
#         model.get_emb_vis(torch.from_numpy(data_descriptors[:, :, ii])).cpu().numpy()
#     )

# ax[-1].imshow(model_descriptors_visualizations)
# fig.tight_layout()

# # %%
# indices = np.zeros(n_objs, dtype=int)
# indices[3] = 80
# initial_trace = Trace(
#     ids=list(range(n_objs)),
#     poses=[
#         cam_pose.dot(transform_matrices[indices[idx]])
#         for idx, transform_matrices in enumerate(pose_hypotheses)
#     ],
#     camera_pose=cam_pose,
# )
# init_depth = gl_renderer.render(initial_trace.ids, initial_trace.poses, cam_pose)[1][
#     :, :, -1
# ]
# fig, ax = plt.subplots(1, 3, figsize=(40, 10))
# ax[0].imshow(test_img.rgb)
# ax[1].imshow(gt_depth)
# ax[1].set_title('GT Depth', fontsize=40)
# ax[2].imshow(init_depth, cmap='turbo')
# ax[2].set_title('MCMC initialization', fontsize=40)
# fig.tight_layout()

# # %% [markdown]
# # ## MCMC inference

# # %%
# log_normalizers = voting.get_max_indices_normalizers_probs(
#     data_descriptors, np.array(bop_obj_indices), squeeze=False
# )[1]

# # %%
# dsize = (gl_renderer.intrinsics.width, gl_renderer.intrinsics.height)
# data_xyz_scaled = cv2.resize(
#     data_xyz,
#     dsize=dsize,
#     interpolation=0,
# )
# log_normalizers = cv2.resize(
#     log_normalizers,
#     dsize=dsize,
#     interpolation=0,
# )
# if len(bop_obj_indices) == 1:
#     log_normalizers = log_normalizers[..., None]

# data_descriptors = np.concatenate(
#     [
#         cv2.resize(
#             data_descriptors[:, :, i, :],
#             dsize=dsize,
#             interpolation=0,
#         )[:, :, None, :]
#         for i in range(data_descriptors.shape[-2])
#     ],
#     axis=-2,
# )


from IPython import embed; embed()