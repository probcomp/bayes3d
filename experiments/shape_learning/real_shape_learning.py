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

sys.path.extend(["/home/nishadgothoskar/ptamp/pybullet_planning"])
sys.path.extend(["/home/nishadgothoskar/ptamp"])
warnings.filterwarnings("ignore")

test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs/spoon_learning.pkl")
file = open(test_pkl_file,'rb')
camera_images = pickle.load(file)["camera_images"]

observations = [jax3dp3.Jax3DP3Observation.construct_from_camera_image(img, near=0.01, far=2.0) for img in camera_images]
print('len(observations):');print(len(observations))

observation = observations[-1]
state = jax3dp3.OnlineJax3DP3()

state.setup_on_initial_frame(observations[-1], [], [])

all_clouds = []
indices = range(len(observations))
self = state
for i in indices:
    observation = observations[i]

    obs_point_cloud_image = self.process_depth_to_point_cloud_image(observation.depth)
    segmentation_image, dashboard_viz  = self.segment_scene(
        observation.rgb, obs_point_cloud_image
    )
    dashboard_viz.save(f"shape_learning_{i}.png")

    unique = jnp.unique(segmentation_image)
    all_seg_ids = unique[unique != -1]

    segment_clouds = []
    dist_to_center = []
    for seg_id in all_seg_ids: 
        cloud = t3d.apply_transform(
            t3d.point_cloud_image_to_points(
                jax3dp3.get_image_masked(obs_point_cloud_image, segmentation_image, seg_id)
            ),
            observation.camera_pose
        )
        segment_clouds += [cloud]
        dist_to_center += [
            jnp.linalg.norm(jnp.mean(cloud, axis=0) - jnp.array([0.5, 0.0, 0.0]))
        ]
    best_cloud = segment_clouds[np.argmin(dist_to_center)]
    # jax3dp3.show_cloud("c1", segment_clouds[1])
    all_clouds.append(best_cloud)


fused_clouds_over_time = [all_clouds[0]]
for i in range(1, len(all_clouds)):
    fused_cloud = fused_clouds_over_time[-1]
    best_transform = jax3dp3.icp.icp_open3d(all_clouds[i], fused_cloud)
    fused_cloud = jnp.vstack(
        [
            fused_cloud,
            t3d.apply_transform(all_clouds[i], best_transform)
        ]
    )
    fused_clouds_over_time.append(fused_cloud)