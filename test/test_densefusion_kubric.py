import sys
densefusion_path = "../jax3dp3/posecnn-pytorch/PoseCNN-PyTorch"
sys.path.append(densefusion_path)   # TODO cleaner import / add to path
import os
import easydict
import pickle
import torch 
import numpy as np
import jax
import jax.numpy as jnp
import random
import trimesh
import bayes3d.posecnn_densefusion
import tools._init_paths
from tools.test_images_utils import env_setup_posecnn, get_image_posecnn, run_posecnn, get_image_densefusion, env_setup_densefusion, run_DenseFusion
from fcn.config import cfg, cfg_from_file

import bayes3d as j


densefusion = j.posecnn_densefusion.DenseFusion()


# --- creating the model dir from the working directory
model_dir = os.path.join(j.utils.get_assets_dir(), "ycb_video_models/models")
print(f"{model_dir} exists: {os.path.exists(model_dir)}")
model_names = j.ycb_loader.MODEL_NAMES

camera_pose = j.t3d.transform_from_pos_target_up(
    jnp.array([0.0, 0.5, 0.0]),
    jnp.array([0.0, 0.0, 0.0]),
    jnp.array([0.0, 0.0, 1.5]),
)

bop_ycb_dir = os.path.join(j.utils.get_assets_dir(), "bop/ycbv")
rgbd, _, _, _ = j.ycb_loader.get_test_img('52', '1', bop_ycb_dir)
intrinsics = j.Intrinsics(
    height=rgbd.intrinsics.height,
    width=rgbd.intrinsics.width,
    fx=rgbd.intrinsics.fx, fy=rgbd.intrinsics.fx,
    cx=rgbd.intrinsics.width/2.0, cy=rgbd.intrinsics.height/2.0,
    near=0.001, far=3.0
)

IDX = 13
NUM_IMAGES = 1

key = jax.random.PRNGKey(0)
object_poses = jax.vmap(lambda key: j.distributions.gaussian_vmf(key, 0.00001, 0.001))(
    jax.random.split(key, NUM_IMAGES)
)
object_poses = jnp.einsum("ij,ajk",j.t3d.inverse_pose(camera_pose),object_poses)

mesh_paths = []
mesh_path = os.path.join(model_dir,model_names[IDX],"textured.obj")
for _ in range(NUM_IMAGES):
    mesh_paths.append(mesh_path)
_, offset_pose = j.mesh.center_mesh(trimesh.load(mesh_path), return_pose=True)

rgbds = j.kubric_interface.render_multiobject_parallel(mesh_paths, object_poses[None,:,...], intrinsics, scaling_factor=1.0, lighting=5.0) # multi img singleobj
j.hvstack_images([j.get_rgb_image(r.rgb) for r in rgbds], 1, NUM_IMAGES).save(f"kubric_densefusion_dataset.png")
gt_poses = object_poses @ offset_pose

for i, rgbd in enumerate(rgbds):
    pose = gt_poses[i]
    results = densefusion.get_densefusion_results(rgbd.rgb, rgbd.depth, rgbd.intrinsics, scene_name=f"kubric_densefusion_{i}")
print(results)

from IPython import embed; embed()

