import os
import glob
import pickle
import numpy as np
from jax3dp3.segment_scene import get_segmentation




# TODO add tests for two parts of segmentation module (bg removal & segmentation)

example_images_dir = '/home/ubuntu/jax3dp3/assets/panda_dataset_ycb_format/panda_data_pik3/' # wherever the PROCESSED pickle files are stored 
pkl_image_files = []

test_names = ['strawberry_error', 'demo2_nolight', 'knife_sim']   # the full name should look like strawberry_error-0.pik
for test_name in test_names:
    pkl_image_files.extend(sorted(glob.glob(example_images_dir + f'{test_name}*.pkl')))



index_images = range(len(pkl_image_files))

for i in index_images:
    if os.path.exists(pkl_image_files[i]):
        print('\n----------------------')
        scene_name = pkl_image_files[i].split("/")[-1].split(".")[0]

        # read sample
        with open(pkl_image_files[i], 'rb') as f:
            data = pickle.load(f)

        rgba_array = data['rgb'] 
        depth_array = data['depth']
        intrinsics = data['intrinsics']

        segmentation_out = get_segmentation(rgba_array, depth_array, intrinsics, scene_name, factor_depth=1)

        print(f"{np.unique(segmentation_out)} objects in {scene_name}")
        from IPython import embed; embed()

    else:
        print('files not exist %s' % (pkl_image_files[i]))