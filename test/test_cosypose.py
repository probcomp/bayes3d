import os

import bayes3d as b
import numpy as np
from bayes3d.neural.cosypose_baseline import cosypose_utils

bop_ycb_dir = os.path.join(b.utils.get_assets_dir(), "bop/ycbv")
rgbd, gt_ids, gt_poses, masks = b.utils.ycb_loader.get_test_img(
    "55", "1592", bop_ycb_dir
)

pred = cosypose_utils.cosypose_interface(
    np.array(rgbd.rgb), b.K_from_intrinsics(rgbd.intrinsics)
)
