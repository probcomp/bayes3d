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

all_descriptors = jnp.array([i[0] for i in all_descriptors_and_normalizers])
all_normalizers = jnp.array([i[1] for i in all_descriptors_and_normalizers])

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
outlier_prob = 0.1
def likelihood(poses, obs, data_descriptors, log_normalizers):
    model_xyz, obj_ids, obj_coords = render_planes_multiobject_augmented(
        poses, shape_planes, shape_dims, h,w, fx,fy, cx,cy, jnp.array(offsets), jnp.array(scales)
    )
    model_xyz = model_xyz[:,:,:3]
    model_mask = model_xyz[:,:,2] > 0.0
    obj_coords = obj_coords[:,:,:3]
    obs = obs[:,:,:3]

    model_descriptors = jax.vmap(infer_mlp,in_axes=(0,0))(obj_coords.reshape(-1,3), obj_ids.reshape(-1,1))[:,0,0,0,0,:].reshape(120,160,12)

    # log_prob = threedp3_likelihood(obs / 100.0, model_xyz / 100.0, r, outlier_prob)
    log_prob = neural_descriptor_likelihood(
        data_xyz=obs / 100.0,
        data_descriptors=data_descriptors,
        log_normalizers=log_normalizers,
        model_xyz=model_xyz / 100.0,
        model_descriptors=model_descriptors,
        model_mask=model_mask.astype(float),
        obj_ids=obj_ids.astype(float),
        data_mask=obs[:,:,2] > 0.0,
        r=r,
        outlier_prob=outlier_prob,
        filter_shape=(5,5),
    )
    return log_prob

# def likelihood(x, obs):
#     rendered_image = render_from_pose(x) / 100.0
#     weight = threedp3_likelihood(obs / 100.0, rendered_image, r, outlier_prob)
#     return weight



likelihood_parallel = jax.vmap(likelihood, in_axes = (0, None, None, None))
likelihood_parallel_jit = jax.jit(likelihood_parallel)

gt_poses = [np.array(initial_poses_estimates[0]),np.array(initial_poses_estimates[1])]
poses = gt_poses

num_particles = 300
particles = []
for _ in range(num_particles):
    particles.append(np.array(gt_poses))
particles = jnp.stack(particles)


keys = jax.random.split(jax.random.PRNGKey(3), particles.shape[0])

all_particles = []
for t in range(len(ground_truth_images)):
    print("here")
    gt_image, descriptors, normalizers = ground_truth_images[t], all_descriptors[t], all_normalizers[t]
    drift_poses = jax.vmap(gaussian_pose, in_axes=(0, None))(keys, jnp.diag(jnp.array([50.0, 25.0, 0.001])))
    i = 1
    particles = particles.at[:, i].set(jnp.einsum("...ij,...jk->...ik", drift_poses, particles[:,i]))
    weights = likelihood_parallel_jit(particles, gt_image,  descriptors, normalizers)
    parent_idxs = jax.random.categorical(keys[0], weights, shape=weights.shape)
    particles = particles[parent_idxs]
    keys = jax.random.split(keys[0], weights.shape[0])
    all_particles.append(particles.copy())


x = jnp.array(all_particles)

middle_width = 20
top_border = 100
cm = plt.get_cmap("turbo")
all_images = []
min_depth = 10.0
max_depth = 2000.0

render_from_pose_jit = jax.jit(lambda poses: render_planes_multiobject_augmented(
    poses, shape_planes, shape_dims, h,w, fx,fy, cx,cy, jnp.array(offsets), jnp.array(scales)
)[0]
)

for i in range(ground_truth_images.shape[0]):
    rgb = rgb_imgs[i]
    rgb_img = Image.fromarray(
        rgb.astype(np.int8), mode="RGBA"
    )

    depth_img = get_depth_image(ground_truth_images[i,:,:,2],min=min_depth,max=max_depth).resize((original_width,original_height))

    pose = x[i,0,:,:,:]
    rendered_image = render_from_pose_jit(pose)

    rendered_depth_img = get_depth_image(rendered_image[:, :, 2],min=min_depth,max=max_depth).resize((original_width,original_height))

    i1 = rendered_depth_img.copy()
    i2 = rgb_img.copy()
    i1.putalpha(128)
    i2.putalpha(128)
    overlay_img = Image.alpha_composite(i1, i2)

    translations = x[i, :, 1, :3, -1]
    img = render_cloud_at_pose(translations, jnp.eye(4), h, w, fx,fy, cx,cy, 0)
    projected_particles_img = get_depth_image(img[:,:,2] > 0.0, max=1.0).resize((original_width,original_height))


    images = [rgb_img, depth_img, rendered_depth_img, overlay_img, projected_particles_img]
    labels = ["RGB Image Frame {}".format(i), "Depth Image", "Inferred Depth", "Overlay",  "Particles"]
    dst = multi_panel(images, labels, middle_width, top_border, 40)
    all_images.append(dst)

all_images[0].save(
    fp="out_{}_particles.gif".format(num_particles),
    format="GIF",
    append_images=all_images,
    save_all=True,
    duration=100,
    loop=0,
)

from IPython import embed; embed()
