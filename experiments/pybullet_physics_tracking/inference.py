import argparse
import os
from functools import partial

import cv2
import joblib
import numpy as np
import pandas as pd
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
from objects3d.likelihood.ndl import neural_descriptor_likelihood
from objects3d.surfemb.siren_jax import siren_jax_from_surfemb, siren_jax_from_params
from objects3d.bop.data import BOPTestImage
from jax3dp3.rendering_augmented import render_planes_multiobject_augmented
import numpy as np
import os
import jax.numpy as jnp
import jax
import trimesh
from jax3dp3.rendering import render_planes
from jax3dp3.likelihood import threedp3_likelihood
from jax3dp3.utils import (
    make_centered_grid_enumeration_3d_points,
)
from jax3dp3.transforms_3d import quaternion_to_rotation_matrix, depth_to_point_cloud_image
from jax3dp3.distributions import gaussian_vmf, gaussian_vmf_cov, vmf, gaussian_pose
from jax3dp3.shape import get_cube_shape, get_rectangular_prism_shape
from jax3dp3.enumerations_procedure import enumerative_inference_single_frame
from jax3dp3.viz import save_depth_image, get_depth_image, multi_panel
import time
from PIL import Image
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2
from jax.scipy.special import logsumexp
from jax3dp3.viz import multi_panel
from jax3dp3.enumerations import make_grid_enumeration, make_translation_grid_enumeration
from jax3dp3.rendering import render_spheres, render_cloud_at_pose,render_planes_multiobject
from jax3dp3.rendering import render_planes_multiobject
from jax3dp3.utils import get_assets_dir
import jax3dp3
import jax3dp3.utils
import jax3dp3.viz
import jax3dp3.transforms_3d as t3d
from objects3d.bop.detector_crops import get_hypotheses_from_detector_crops


data_npz = np.load("data.npz")
depth_imgs = np.array(data_npz["depth_imgs"]).copy()
rgb_imgs = np.array(data_npz["rgb_imgs"]).copy()

scaling_factor = 0.25

fx = data_npz["fx"] * scaling_factor
fy = data_npz["fy"] * scaling_factor

cx = data_npz["cx"] * scaling_factor
cy = data_npz["cy"] * scaling_factor

original_height = data_npz["height"]
original_width = data_npz["width"]

h = int(np.round(original_height  * scaling_factor))
w = int(np.round(original_width * scaling_factor))
print(h,w,fx,fy,cx,cy)

coord_images = [depth_to_point_cloud_image(cv2.resize(d.copy(), (w,h),interpolation=0), fx,fy,cx,cy) for d in depth_imgs]
rgb_images = [cv2.resize(d.copy(), (w,h),interpolation=0) for d in rgb_imgs]
ground_truth_images = np.stack(coord_images)
ground_truth_images[ground_truth_images[:,:,:,2] > 1900.0] = 0.0
# ground_truth_images[ground_truth_images[:,:,:,1] > 0.85,:] = 0.0
ground_truth_images = np.concatenate([ground_truth_images, np.ones(ground_truth_images.shape[:3])[:,:,:,None] ], axis=-1)
ground_truth_images = jnp.array(ground_truth_images)

import joblib
(
    all_descriptors_and_normalizers,
    all_params,
    offsets,
    scales
) = joblib.load("surfemb_data.joblib")

infer_mlp = siren_jax_from_params(all_params)


shape_filenames = [
    os.path.join(jax3dp3.utils.get_assets_dir(), "models/003_cracker_box/textured_simple.obj"),
    os.path.join(jax3dp3.utils.get_assets_dir(), "models/004_sugar_box/textured_simple.obj")
]
shape_planes = []
shape_dims = []
for f in shape_filenames:
    mesh = trimesh.load(f)
    half_dims = np.array(mesh.vertices).max(0)
    print(half_dims)
    plane, dims = shape = get_rectangular_prism_shape(half_dims*2.0)
    shape_planes.append(plane)
    shape_dims.append(dims)

shape_dims = jnp.array(shape_dims)
shape_planes = jnp.array(shape_planes)

cam_pose = np.eye(4)
cam_pose[:3,0] = np.array([0.0, 1.0, 0.0])
cam_pose[:3,1] = np.array([0.0, 0.0, -1.0])
cam_pose[:3,2] = np.array([-1.0, 0.0, 0.0])
cam_pose[:3,3] = np.array([1000.0, 0.0, 0.0])
cam_pose = jnp.array(cam_pose)

initial_poses_estimates = jnp.array(
    [
        [
            [1.0, 0.0, 0.0, 0.00],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        [
            [1.0, 0.0, 0.0, -30.00],
            [0.0, 1.0, 0.0, -200.00],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    ]
)
initial_poses_estimates = jnp.einsum("ij,ajk->aik", jnp.linalg.inv(cam_pose), initial_poses_estimates)

r=0.1
outlier_prob = 0.000001
def get_score(poses, data_xyz, data_descriptors, log_normalizers):
    model_xyz, obj_ids, obj_coords = render_planes_multiobject_augmented(
        poses, shape_planes, shape_dims, h,w, fx,fy, cx,cy, jnp.array(offsets), jnp.array(scales)
    )
    obj_ids = obj_ids.astype(jnp.int32)
    model_xyz = model_xyz[:,:,:3]
    model_mask = model_xyz[:,:,2] > 0.0
    obj_coords = obj_coords[:,:,:3]

    model_descriptors = jax.vmap(infer_mlp,in_axes=(0,0))(obj_coords.reshape(-1,3), obj_ids.reshape(-1,1))[:,0,0,0,0,:].reshape(120,160,12)
    p_background = outlier_prob
    p_foreground = (1.0 - outlier_prob) / model_mask.sum()

    log_prob = neural_descriptor_likelihood(
        data_xyz=data_xyz / 100.0,
        data_descriptors=data_descriptors,
        log_normalizers=log_normalizers,
        model_xyz=model_xyz / 100.0,
        model_descriptors=model_descriptors,
        model_mask=model_mask.astype(float),
        obj_ids=obj_ids.astype(float),
        data_mask=data_xyz[:,:,2] > 0.0,
        r=r,
        p_background=p_background,
        p_foreground=p_foreground,
        filter_shape=(5,5),
    )
    return log_prob

get_score_jit = jax.jit(get_score)
get_score_parallel_jit = jax.jit(jax.vmap(get_score,in_axes=(0,None, None, None)))
get_score_parallel = jax.vmap(get_score,in_axes=(0,None, None, None))

gt_poses = [np.array(initial_poses_estimates[0]),np.array(initial_poses_estimates[1])]
poses = gt_poses


num_particles = 300
particles = []
for _ in range(num_particles):
    particles.append(np.array(gt_poses))
particles = jnp.stack(particles)
for (t,_) in enumerate(all_descriptors_and_normalizers):
    data_descriptors = all_descriptors_and_normalizers[t][0]
    log_normalizers = all_descriptors_and_normalizers[t][1]
    keys = jax.random.split(jax.random.PRNGKey(3), particles.shape[0])
    drift_poses = jax.vmap(gaussian_pose, in_axes=(0, None))(keys, jnp.diag(jnp.array([50.0, 25.0, 0.001])))
    i = 1
    particles = particles.at[:, i].set(jnp.einsum("...ij,...jk->...ik", drift_poses, particles[:,i]))

    scores = get_score_parallel_jit(particles, jnp.array(coord_images[t]), jnp.array(data_descriptors), jnp.array(log_normalizers))

    # scores = []
    # for iters in range(10):
    #     scores.append(
    #         get_score_parallel_jit(particles[iters*100:iters*100+100], jnp.array(coord_images[t]), jnp.array(data_descriptors), jnp.array(log_normalizers))
    #     )
    # scores = jnp.concatenate(scores)
    parent_idxs = jax.random.categorical(keys[0], scores, shape=scores.shape)
    particles = particles[parent_idxs]
    keys = jax.random.split(keys[0], keys.shape[0])
    print(particles[:,1,0,3])


from IPython import embed; embed()
