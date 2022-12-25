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
from objects3d.likelihood.ndl import neural_descriptor_likelihood
from objects3d.surfemb.siren_jax import siren_jax_from_surfemb
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

from objects3d.likelihood.ndl import JAXNDL

ti.init(arch=ti.vulkan)
device = torch.device('cuda:0')

data_directory = os.path.expanduser('~/data')
surfemb_model_path = os.path.join(data_directory, 'models/ycbv-jwpvdij1.compact.ckpt')
dataset = 'ycbv'


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

data_npz = np.load("data.npz")
depth_imgs = np.array(data_npz["depth_imgs"]).copy()
rgb_imgs = np.array(data_npz["rgb_imgs"]).copy()

min_depth = 900.0
max_depth = 1200.0
middle_width = 20
top_border = 100
cm = plt.get_cmap("turbo")

scaling_factor = 1.0

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


r = 0.1
outlier_prob = 0.01

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


intrinsics = np.array([
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0]
])
cam_pose = np.eye(4)

bop_obj_indices = [2,3]

all_test_images = [
    BOPTestImage(
    dataset='ycbv',
    scene_id=-1,
    img_id=-1,
    rgb=rgb_images[image_index][:,:,:3],
    depth=coord_images[image_index][:,:,2],
    intrinsics=intrinsics,
    camera_pose=cam_pose,
    bop_obj_indices=tuple(bop_obj_indices),
    default_scales=np.array(
        [1.5, 1.5]
    ),
    annotations=None
    )
    for image_index in range(len(rgb_images))
]   

gl_renderer = all_test_images[0].get_renderer(data_directory, 1.0)
intrinsics = all_test_images[0].intrinsics.copy()
n_objs = len(bop_obj_indices)





r = 0.1
outlier_prob = 0.000001

from IPython import embed; embed()

gt_poses = [np.array(initial_poses_estimates[0]),np.array(initial_poses_estimates[1])]
poses = gt_poses

model_xyz, obj_ids, obj_coords = render_planes_multiobject_augmented(
    jnp.array(poses), shape_planes, shape_dims, h,w, fx,fy, cx,cy
)
a,b = 200,120
point_on_object = obj_coords[a,b]
id = obj_ids[a,b]

print(model_xyz[200,120,:])
save_depth_image(model_xyz[:,:,2], "1.png", max=2000.0)

model_xyz_gl, obj_coords_gl, model_mask_gl, obj_ids_gl = gl_renderer.render(
    [0,1], poses
)
print(model_xyz_gl[200,120,:])
save_depth_image(model_xyz_gl[:,:,2], "2.png", max=2000.0)

def get_score(poses, data_xyz, data_descriptors, log_normalizers):
    model_xyz, obj_ids, obj_coords = render_planes_multiobject_augmented(
        jnp.array(poses), shape_planes, shape_dims, h,w, fx,fy, cx,cy
    )
    model_xyz = model_xyz[:,:,:3]
    obj_coords = obj_coords[:,:,:3]
    obj_ids = np.array(obj_ids)
    save_depth_image(model_xyz[:,:,2], "1.png", max=2000.0)
    save_depth_image(np.array(obj_ids) + 1.0, "1.png", max=3.0)

    # model_xyz, obj_coords, model_mask, obj_ids = gl_renderer.render(
    #     [0,1], poses
    # )
    # save_depth_image(np.array(obj_ids) + 1.0, "2.png", max=3.0)
    # save_depth_image(model_xyz[:,:,2], "2.png", max=2000.0)

    obj_coords = torch.from_numpy(np.array(obj_coords).astype(np.float32)).to(model.device)
    model_descriptors = np.zeros(model_mask.shape + (model.emb_dim,))
    for ii in range(len(gl_renderer.bop_obj_indices)):
        model_descriptors[obj_ids == ii] = (
            model.infer_mlp(
                obj_coords,
                gl_renderer.bop_obj_indices[ii],
            )[obj_ids == ii]
            .cpu()
            .numpy()
        )
    p_background = outlier_prob
    p_foreground = (1.0 - outlier_prob) / model_mask.sum()

    log_prob = neural_descriptor_likelihood(
        data_xyz=data_xyz,
        data_descriptors=data_descriptors,
        log_normalizers=log_normalizers,
        model_xyz=model_xyz,
        model_descriptors=model_descriptors,
        model_mask=model_mask.astype(float),
        obj_ids=obj_ids.astype(float),
        data_mask=data_xyz[:,:,2] > 0.0,
        r=r,
        p_background=p_background,
        p_foreground=p_foreground,
        filter_shape=(5,5),
    )
    return log_prob.item()

for (t,test_img) in enumerate(all_test_images):
    data_descriptors = surfemb.get_data_descriptors(
        test_img,
        1.5,
        bop_obj_indices=bop_obj_indices,
        use_cpu=True
    )
    log_normalizers = voting.get_max_indices_normalizers_probs(
        data_descriptors, np.array(bop_obj_indices), squeeze=False
    )[1]

    fig, ax = plt.subplots(
        1,
        len(test_img.bop_obj_indices) + 1,
        figsize=(10 * (len(test_img.bop_obj_indices) + 1), 15),
    )
    for ii in range(len(test_img.bop_obj_indices)):
        ax[ii].imshow(
            model.get_emb_vis(torch.from_numpy(data_descriptors[:, :, ii])).cpu().numpy()
        )
    fig.tight_layout()
    fig.savefig("descriptors_{}.png".format(t))

    coords =  np.linspace(-400.0,200.0,100)
    values = []
    for i in coords:
        poses = gt_poses.copy()
        poses[1][0,3] = i
        score = get_score(poses, coord_images[t], data_descriptors, log_normalizers)
        values.append(score)
    print(coords[np.argmax(values)])






from IPython import embed; embed()