import sys

densefusion_path = "../jax3dp3/posecnn-pytorch/PoseCNN-PyTorch"
sys.path.append(densefusion_path)   # TODO cleaner import / add to path
import os
import easydict
import pickle
import torch 
import numpy as np
import random
import tools._init_paths
from tools.test_images_utils import env_setup_posecnn, get_image_posecnn, run_posecnn, get_image_densefusion, env_setup_densefusion, run_DenseFusion
from fcn.config import cfg, cfg_from_file

import jax3dp3 as j

# Input here
test_filename_pik = '/home/ubuntu/jax3dp3/jax3dp3/posecnn-pytorch/PoseCNN-PyTorch/datasets/pandas/data/new_panda/demo2_nolight-0.pkl'
test_filename_pik = '/home/ubuntu/jax3dp3/jax3dp3/posecnn-pytorch/PoseCNN-PyTorch/datasets/pandas/data/new_panda/strawberry_error-0.pkl'
test_filename_pik = '/home/ubuntu/jax3dp3/jax3dp3/posecnn-pytorch/PoseCNN-PyTorch/datasets/pandas/data/new_panda/knife_sim-0.pkl'
scene_name = test_filename_pik.split('/')[-1].split('.')[0]
print(scene_name)


# Load Test Image
with open(test_filename_pik, 'rb') as file:
    test_data = pickle.load(file)
_, image_depth, meta_data = get_image_posecnn(test_data)  
image_color_rgb, _, _ = get_image_densefusion(test_data)
image_color_rgb = image_color_rgb[:,:,:3]


K = test_data['intrinsics']
intrinsics = j.Intrinsics(
    height=300,
    width=300,
    fx=K[0][0], fy=K[1][1],
    cx=K[0][-1], cy=K[1][-1],
    near=0.001, far=50.0
)

results = j.posecnn_densefusion.get_densefusion_results(image_color_rgb, image_depth, intrinsics, scene_name=scene_name)

print(results)


from IPython import embed; embed()
