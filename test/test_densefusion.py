import sys
import os
import easydict
import pickle
import torch 
import numpy as np
import random
import bayes3d.posecnn_densefusion

import bayes3d as j


bop_ycb_dir = os.path.join(j.utils.get_assets_dir(), "bop/ycbv")
rgbd, gt_ids, gt_poses, masks = j.ycb_loader.get_test_img('48', '1', bop_ycb_dir)

import bayes3d.posecnn_densefusion
densefusion = bayes3d.posecnn_densefusion.DenseFusion()
results = densefusion.get_densefusion_results(rgbd.rgb, rgbd.depth, rgbd.intrinsics, scene_name="1")
print(results)

rgbd, gt_ids, gt_poses, masks = j.ycb_loader.get_test_img('55', '22', bop_ycb_dir)
results = densefusion.get_densefusion_results(rgbd.rgb, rgbd.depth, rgbd.intrinsics, scene_name="1")
print(results)

from IPython import embed; embed()
