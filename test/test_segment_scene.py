import os
import glob
import pickle
import numpy as np
import bayes3d.segment_scene
import bayes3d
import time



test_pkl_file = os.path.join(bayes3d.utils.get_assets_dir(),"sample_imgs/cracker_sugar_banana_real.pkl")
test_pkl_file = os.path.join(bayes3d.utils.get_assets_dir(),"sample_imgs/red_lego_multi.pkl")
test_pkl_file = os.path.join(bayes3d.utils.get_assets_dir(),"sample_imgs/red_lego_multi.pkl")
test_pkl_file = os.path.join(bayes3d.utils.get_assets_dir(),"sample_imgs/strawberry_error.pkl")
test_pkl_file = os.path.join(bayes3d.utils.get_assets_dir(),"sample_imgs/utensils.pkl")
test_pkl_file = os.path.join(bayes3d.utils.get_assets_dir(),"sample_imgs/knife_spoon_box_real.pkl")
# test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs/demo2_nolight.pkl")
with open(test_pkl_file, 'rb') as f:
    camera_images = pickle.load(f)["camera_images"]

camera_image = camera_images[-1]
rgba_array = camera_image.rgbPixels 
depth_array = camera_image.depthPixels
intrinsics = camera_image.camera_matrix

observation = bayes3d.Jax3DP3Observation.construct_from_camera_image(camera_image, near=0.01, far=2.0)
(h,w,fx,fy,cx,cy,near,far) = observation.camera_params

foreground_mask = bayes3d.segment_scene.get_foreground_mask(rgba_array)

start = time.time()
foreground_mask = bayes3d.segment_scene.get_foreground_mask(rgba_array)
print(time.time() - start)

rgb_viz = bayes3d.viz.get_rgb_image(rgba_array, 255.0)
rgb_viz_masked = bayes3d.viz.get_rgb_image(rgba_array* foreground_mask[..., None] , 255.0)
bayes3d.viz.multi_panel([rgb_viz, rgb_viz_masked], labels=["RGB", "RGB Masked"]).save("out.png")

foreground_mask, segmentation_out = bayes3d.segment_scene.get_segmentation(rgba_array, depth_array, fx,fy,cx,cy)

start = time.time()
foreground_mask, segmentation_out = bayes3d.segment_scene.get_segmentation(rgba_array, depth_array, fx,fy,cx,cy)
print(time.time() - start) 

rgb_viz = bayes3d.viz.get_rgb_image(rgba_array, 255.0)
rgb_viz_masked = bayes3d.viz.get_rgb_image(rgba_array* foreground_mask[..., None] , 255.0)
depth_viz = bayes3d.viz.get_depth_image(depth_array, max=far)
seg_viz = bayes3d.viz.get_depth_image(segmentation_out + 1, max=4.0)
bayes3d.viz.multi_panel([rgb_viz, rgb_viz_masked, depth_viz, seg_viz], labels=["RGB", "RGB Masked", "Depth", "Segmentation"]).save("out.png")

from IPython import embed; embed()