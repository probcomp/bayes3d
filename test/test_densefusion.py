import sys
densefusion_path = "../jax3dp3/posecnn-pytorch/PoseCNN-PyTorch"
sys.path.append(densefusion_path)   # TODO cleaner import / add to path
import os
import easydict
import pickle
import torch 
import numpy as np
import random
import jax3dp3.posecnn_densefusion
import tools._init_paths
from tools.test_images_utils import env_setup_posecnn, get_image_posecnn, run_posecnn, get_image_densefusion, env_setup_densefusion, run_DenseFusion
from fcn.config import cfg, cfg_from_file

import jax3dp3 as j


bop_ycb_dir = os.path.join(j.utils.get_assets_dir(), "bop/ycbv")
rgbd, gt_ids, gt_poses, masks = j.ycb_loader.get_test_img('52', '1', bop_ycb_dir)

densefusion = j.posecnn_densefusion.DenseFusion()
results = densefusion.get_densefusion_results(rgbd.rgb, rgbd.depth, rgbd.intrinsics, scene_name="1")
print(results)

rgbd, gt_ids, gt_poses, masks = j.ycb_loader.get_test_img('55', '22', bop_ycb_dir)
results = densefusion.get_densefusion_results(rgbd.rgb, rgbd.depth, rgbd.intrinsics, scene_name="1")
print(results)

from IPython import embed; embed()
