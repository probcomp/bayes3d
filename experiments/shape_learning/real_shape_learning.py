import numpy as np
import os
import jax
import jax.numpy as jnp
import trimesh
import time
import pickle
import jax3dp3.transforms_3d as t3d
import jax3dp3 as j
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

test_pkl_file = os.path.join(j.utils.get_assets_dir(),"sample_imgs/spoon_learning.pkl")
file = open(test_pkl_file,'rb')
camera_images = pickle.load(file)["camera_images"]

images = [j.RGBD.construct_from_camera_image(img, near=0.001, far=2.0) for img in camera_images]

image = images[0]

from IPython import embed; embed()

import jax3dp3.segment_scene
obs_point_cloud_image = j.t3d.unproject_depth(image.depth, image.intrinsics)
segmentation_image, mask, viz = jax3dp3.segment_scene.segment_scene(
    image.rgb,
    obs_point_cloud_image,
    image.intrinsics
)
viz.save("viz.png")

cloud = j.t3d.apply_transform(obs_point_cloud_image[ segmentation_image == 0], image.camera_pose)
j.meshcat.setup_visualizer()
j.meshcat.show_cloud("1",cloud)

dims, pose = j.utils.aabb(cloud)
grid = j.make_translation_grid_enumeration_3d(
    *np.array(pose[:3,3] - dims/2.0), *np.array(pose[:3,3] + dims/2.0),
    50,50,50
)


depth_images = []
camera_poses = []
for image in images:
    depth_images.append(image.depth)
    camera_poses.append(image.camera_pose)
depth_images = jnp.array(depth_images)
camera_poses = jnp.array(camera_poses)

intrinsics = image.intrinsics

j.viz.get_rgb_image(image.rgb).save("rgb.png")

t=0
threshold = 0.005
depth_images_modified = depth_images.at[depth_images <intrinsics.near].set(intrinsics.far)

counts_positive = jnp.zeros(grid.shape[0])
counts_negative = jnp.zeros(grid.shape[0])

assignment = j.occlusion.voxel_occupied_occluded_free_jit(
    camera_poses[t], depth_images_modified[t], grid, intrinsics,threshold
)
counts_positive = counts_positive.at[assignment > 0.5].set(counts_positive[assignment > 0.5] + 1.0)
counts_negative = counts_negative.at[assignment < 0.5].set(counts_negative[assignment < 0.5] + 1.0)

pseudocounts = 0.01
shape_distrib = (counts_positive + pseudocounts)/(counts_negative + counts_positive + 2*pseudocounts)

j.meshcat.clear()
j.meshcat.show_cloud("occupied", grid[shape_distrib > 0.5])
j.meshcat.show_cloud("occl", grid[shape_distrib == 0.5], color=j.RED)


translation_deltas = j.make_translation_grid_enumeration(
    -0.1, -0.1, -0.1, 0.1, 0.1, 0.1, 5,5,5
)

t = 1
actual_camera_pose = camera_poses[t]
proposals = jnp.einsum("ij,ajk->aik", actual_camera_pose, translation_deltas)

assignments = j.occlusion.voxel_occupied_occluded_free_parallel_camera(
    proposals, depth_images_modified[t], grid, intrinsics, threshold
)
i = 1
j.meshcat.clear()
j.meshcat.show_cloud("occupied", grid[assignments[i] > 0.5])
j.meshcat.show_cloud("occl", grid[assignments[i] == 0.5], color=j.RED)
j.meshcat.show_pose("p1", proposals[i])
j.meshcat.show_pose("p2",actual_camera_pose)



shape_distrib = (counts_positive + pseudocounts)/(counts_negative + counts_positive + 2*pseudocounts)
probabilities = (assignments == 1.0) * shape_distrib + (assignments == 0.0) * (1.0 - shape_distrib) + (assignments == 0.5)
total_probabilities = jnp.log(probabilities).sum(1)
best_idx = total_probabilities.argmax()

counts_positive = counts_positive.at[assignments[best_idx] > 0.5].set(counts_positive[assignments[best_idx] > 0.5] + 1.0)
counts_negative = counts_negative.at[assignments[best_idx] < 0.5].set(counts_negative[assignments[best_idx] < 0.5] + 1.0)


inferred_assignments = []

for t in range(1, len(depth_images_modified)):
    actual_camera_pose = camera_poses[t]

    proposals = jnp.einsum("ij,ajk->aik", actual_camera_pose, translation_deltas)
    assignments = j.occlusion.voxel_occupied_occluded_free_parallel_camera(
        proposals, depth_images_modified[t], grid, intrinsics, threshold
    )

    shape_distrib = (counts_positive + pseudocounts)/(counts_negative + counts_positive + 2*pseudocounts)
    probabilities = (assignments == 1.0) * shape_distrib + (assignments == 0.0) * (1.0 - shape_distrib) + (assignments == 0.5)
    total_probabilities = jnp.log(probabilities).sum(1)
    best_idx = total_probabilities.argmax()

    counts_positive = counts_positive.at[assignments[best_idx] > 0.5].set(counts_positive[assignments[best_idx] > 0.5] + 1.0)
    counts_negative = counts_negative.at[assignments[best_idx] < 0.5].set(counts_negative[assignments[best_idx] < 0.5] + 1.0)

    inferred_assignments.append(assignments[best_idx])


shape_distrib = (counts_positive + pseudocounts)/(counts_negative + counts_positive + 2*pseudocounts)

j.meshcat.clear()
j.meshcat.show_cloud("occupied", grid[shape_distrib > 0.8] )
j.meshcat.show_cloud("occl", grid[shape_distrib == 0.5] , color=j.RED)


j.meshcat.clear()
i = 3
j.meshcat.show_cloud("occupied", grid[inferred_assignments[i] > 0.8] )
j.meshcat.show_cloud("occl", grid[inferred_assignments[i] == 0.5] , color=j.RED)



assignments = j.occlusion.voxel_occupied_occluded_free_parallel_camera_depth(
    camera_poses, depth_images_modified, grid, intrinsics,threshold
)

j.meshcat.clear()
j.meshcat.show_cloud("occupied", grid[assignments[2] > 0.5])