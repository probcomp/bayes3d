
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

test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs/demo2_nolight.pkl")
test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs/red_lego_multi.pkl")
test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs/knife_spoon_box_real.pkl")
test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs/strawberry_error.pkl")
test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs/knife_sim.pkl")
test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs/knife_spoon_real.pkl")
file = open(test_pkl_file,'rb')
camera_images = pickle.load(file)["camera_images"]


file.close()
if type(camera_images) != list:
    camera_images = [camera_images]

observations = [jax3dp3.Jax3DP3Observation.construct_from_camera_image(img, near=0.01, far=2.0) for img in camera_images]
print('len(observations):');print(len(observations))

observation = observations[0]
state = jax3dp3.OnlineJax3DP3()
state.start_renderer(observation.camera_params)

point_cloud_image = state.process_depth_to_point_cloud_image(observations[0].depth)
state.infer_table_plane(point_cloud_image, observations[0].camera_pose)

top_level_dir = os.path.dirname(os.path.dirname(pybullet_planning.__file__))
model_names = ["knife", "spoon", "cracker_box", "strawberry", "mustard_bottle", "sugar_box","banana"]
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
for (mesh, mesh_name) in zip(meshes, model_names):
    state.add_trimesh(mesh, mesh_name)

state.set_coarse_to_fine_schedules(
    grid_widths=[0.1, 0.01, 0.04, 0.02],
    angle_widths=[jnp.pi, jnp.pi, 0.001, jnp.pi/10],
    grid_params=[(7, 7 ,21),(7, 7 ,21),(15, 15, 1), (7, 7 ,21)],
)

obs_point_cloud_image = state.process_depth_to_point_cloud_image(observation.depth)
segmentation_image, dashboard_viz = state.segment_scene(observation.rgb, obs_point_cloud_image)
dashboard_viz.save("dashboard_1.png")


seg_id = 0.0
hypotheses_over_time, inference_viz = state.inference_for_segment(
    observation.rgb,
    obs_point_cloud_image,
    segmentation_image,
    seg_id,
    [0,1],
    # jnp.arange(len(state.meshes)),
    observation.camera_pose, 
    jnp.array([0.01, 0.005, 0.001]),
    outlier_prob=0.2,
    outlier_volume=1.0
)
inference_viz.save("predictions.png")

from IPython import embed; embed()

exact_match_score = jax3dp3.threedp3_likelihood_parallel_jit(
    obs_point_cloud_image, jnp.array([obs_point_cloud_image]), 0.0005, 0.2, 1.0
)[0]
final_scores = jnp.array([i[0] for i in hypotheses_over_time[-1]])
known_object_score = (jnp.array(final_scores) - exact_match_score) / ((segmentation_image == seg_id).sum()) * 1000.0
print(known_object_score)






