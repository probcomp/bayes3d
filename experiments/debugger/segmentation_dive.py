
import numpy as np
import os
import jax
import jax.numpy as jnp
import trimesh
import time
import pickle
import jax3dp3.transforms_3d as t3d
import jax3dp3
from dataclasses import dataclass
import sys
import warnings
import pybullet_planning
import cv2
import collections
import heapq

sys.path.extend(["/home/ubuntu/ptamp/pybullet_planning"])
sys.path.extend(["/home/ubuntu/ptamp"])
warnings.filterwarnings("ignore")

test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs/red_lego_multi.pkl")
test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs/knife_sim.pkl")
test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs/strawberry_error.pkl")
test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs/demo2_nolight.pkl")
test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs/knife_spoon_real.pkl")
test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs/knife_spoon_box_real.pkl")
test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs/utensils.pkl")
test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs/red_lego.pkl")
# test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs/cracker_sugar_banana_real.pkl")
test_pkl_name = test_pkl_file.split(".")[0].split("/")[-1]

file = open(test_pkl_file,'rb')
camera_images = pickle.load(file)["camera_images"]


file.close()
if type(camera_images) != list:
    camera_images = [camera_images]

observations = [jax3dp3.Jax3DP3Observation.construct_from_camera_image(img, near=0.01, far=2.0) for img in camera_images]
print('len(observations):');print(len(observations))

observation = observations[0]
state = jax3dp3.OnlineJax3DP3()

top_level_dir = os.path.dirname(os.path.dirname(pybullet_planning.__file__))
mesh_names = ["knife", "spoon", "cracker_box", "strawberry", "mustard_bottle", "sugar_box","banana"]
model_paths = [
    os.path.join(top_level_dir,"models/srl/ycb/032_knife/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/031_spoon/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/003_cracker_box/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/012_strawberry/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/006_mustard_bottle/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/004_sugar_box/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/011_banana/textured.obj"),
]
meshes = [
    trimesh.load(path) for path in model_paths
]
state.setup_on_initial_frame(observations[0], meshes, mesh_names)


obs_point_cloud_image = state.process_depth_to_point_cloud_image(observation.depth)

mask_array = state.get_foreground_mask(observation.rgb, obs_point_cloud_image)

cluster_image, dashboard_viz = state.cluster_scene_from_mask(observation.rgb, obs_point_cloud_image, mask_array)
dashboard_viz.save(f"dashboard_cluster_{test_pkl_name}.png")

# from IPython import embed; embed()
segmentation_image, dashboard_viz = state.segment_scene_from_mask(observation.rgb, observation.depth, mask_array)

if segmentation_image is not None:
    dashboard_viz.save(f"dashboard_nn_{test_pkl_name}.png")
else:
    segmentation_image = cluster_image  

num_objects = int(cluster_image.max()) + 1


final_segmentation = np.zeros(cluster_image.shape) - 1
final_cluster_id = 0
for cluster_id in range(num_objects):
    print("\n\n Cluster id = ", cluster_id)

    cluster_region = cluster_image == cluster_id

    cluster_region_nn_pred = segmentation_image[cluster_region]
    cluster_region_nn_pred_items = set(np.unique(cluster_region_nn_pred)) - {-1}

    # from IPython import embed; embed()

    # TODO
    if len(cluster_region_nn_pred_items) == 1:
        print("No further segmentation:cluster id ", final_cluster_id)
        final_segmentation[cluster_region] = final_cluster_id 
        final_cluster_id += 1
    else:  # split region 
        nn_segmentation = segmentation_image[cluster_region]
        final_segmentation[cluster_region] = nn_segmentation - nn_segmentation.min() + final_cluster_id
        print("Extra segmentation: cluster id ", np.unique(final_segmentation[cluster_region]))

        final_cluster_id = final_segmentation[cluster_region].max() + 1

    
final_segmentation_viz = jax3dp3.viz.get_depth_image(final_segmentation + 1, max=final_segmentation.max() + 1)


viz_h, viz_w = final_segmentation.shape
viz_image = jax3dp3.viz.multi_panel(
    [
        jax3dp3.viz.resize_image(jax3dp3.viz.get_rgb_image(observation.rgb, 255.0), viz_h, viz_w),
        jax3dp3.viz.resize_image(jax3dp3.viz.get_rgb_image(observation.rgb * mask_array[:,:,None], 255.0), viz_h, viz_w),
        final_segmentation_viz,
    ],
    labels=["RGB", "RGB Masked", f"Segmentation:{int(final_segmentation.max()) + 1}"]
)


viz_image.save(f"dashboard_final_{test_pkl_name}.png")

from IPython import embed; embed()





# state.step(observation, 1)









