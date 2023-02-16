
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

test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs/red_lego_multi.pkl")
test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs/knife_sim.pkl")
test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs/strawberry_error.pkl")
test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs/demo2_nolight.pkl")
test_pkl_file = os.path.join(jax3dp3.utils.get_assets_dir(),"sample_imgs/knife_spoon_box_real.pkl")


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

state.set_coarse_to_fine_schedules(
    grid_widths=[0.15, 0.01, 0.04, 0.02],
    angle_widths=[jnp.pi, jnp.pi, 0.001, jnp.pi/10],
    grid_params=[(7,7,21),(7,7,21),(15, 15, 1), (7,7,21)],
)

obs_point_cloud_image = state.process_depth_to_point_cloud_image(observation.depth)
segmentation_image, dashboard_viz = state.segment_scene(observation.rgb, obs_point_cloud_image)
dashboard_viz.save("dashboard_1.png")

from IPython import embed; embed()
# state.step(observation, 1)

seg_id = 0.0
r_sweep = jnp.array([0.01])
outlier_prob=0.3
outlier_volume=1.0

hypotheses_over_time, _, inference_viz = state.classify_segment(
    observation.rgb,
    obs_point_cloud_image,
    segmentation_image,
    seg_id,
    # [0,1,2,3,4],
    jnp.arange(len(state.meshes)),
    observation.camera_pose, 
    r_sweep,
    outlier_prob=outlier_prob,
    outlier_volume=outlier_volume
)
inference_viz.save("predictions.png")


obj_idx = 1
good_poses, occlusion_viz = state.occluded_object_search(
    observation.rgb,
    obs_point_cloud_image,
    obj_idx,
    observation.camera_pose, 
    0.01,
    segmentation_image,
    (20,20,4),
    outlier_prob=0.2,
    outlier_volume=1.0
)
occlusion_viz.save("occlusion_1.png")

jax3dp3.setup_visualizer()
jax3dp3.show_cloud("c", t3d.apply_transform(
    t3d.point_cloud_image_to_points(obs_point_cloud_image),
    observation.camera_pose
))

jax3dp3.show_cloud("obj", t3d.apply_transform(
    state.meshes[obj_idx].vertices,
    observation.camera_pose @ good_poses[1]
), color=np.array([1.0, 0.0, 0.0]))

all_imgs = jax3dp3.render_parallel(good_poses, obj_idx)
x = all_imgs[:,:,:,-1].sum(-1).sum(-1)
good_poses = good_poses[x>0]





from IPython import embed; embed()






