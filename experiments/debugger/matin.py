
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

sys.path.extend(["/home/nishadgothoskar/ptamp/pybullet_planning"])
sys.path.extend(["/home/nishadgothoskar/ptamp"])
warnings.filterwarnings("ignore")

# filename = "panda_dataset/scene_4.pkl"
# filename = "panda_dataset_2/utensils.pkl"
# full_filename = "blue.pkl"
full_filename = "1674620488.514845.pkl"
full_filename = "knife_spoon.pkl"
full_filename = "new_utencils.pkl"
# full_filename = os.path.join(jax3dp3.utils.get_assets_dir(), filename)
full_filename = "utensils.pkl"
full_filename = "new_utencils.pkl"
full_filename = "utensils.pkl"
full_filename = "nishad_0.pkl"
full_filename = "shape_acquisition.pkl"

full_filename = "strawberry_error.pkl"

full_filename = "demo2_nolight.pkl"

full_filename = "knife_spoon.pkl"
full_filename = "nishad_1.pkl"
file = open(full_filename,'rb')
camera_images = pickle.load(file)["camera_images"]
# camera_images = pickle.load(file)
file.close()
if type(camera_images) != list:
    camera_images = [camera_images]

observations = [jax3dp3.Jax3DP3Observation.construct_from_camera_image(img) for img in camera_images]
print('len(observations):');print(len(observations))

# jax3dp3.setup_visualizer()

perception_state = jax3dp3.OnlineJax3DP3()
perception_state.set_camera_parameters(observations[0].camera_params, scaling_factor=0.3)

observation = observations[-1]
point_cloud_image = perception_state.process_depth_to_point_cloud_image(observation.depth)
perception_state.infer_table_plane(point_cloud_image, observation.camera_pose)

perception_state.start_renderer()

top_level_dir = os.path.dirname(os.path.dirname(pybullet_planning.__file__))
model_names = ["knife", "spoon", "cracker_box", "strawberry", "mustard_bottle", "sugar_box","banana"]
model_paths = [
    # os.path.join(top_level_dir,"models/srl/ycb/032_knife/textured.obj"),
    # os.path.join(top_level_dir,"models/srl/ycb/031_spoon/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/003_cracker_box/textured.obj"),
    # os.path.join(top_level_dir,"models/srl/ycb/012_strawberry/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/006_mustard_bottle/textured.obj"),
    # os.path.join(top_level_dir,"models/srl/ycb/004_sugar_box/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/011_banana/textured.obj"),
]
for (name, path) in zip(model_names, model_paths):
    perception_state.add_trimesh(
        trimesh.load(path), mesh_name=name, mesh_scaling_factor=1.0
    )


perception_state.set_coarse_to_fine_schedules(
    grid_widths=[0.15, 0.1, 0.07, 0.04, 0.02, 0.01, 0.02],
    grid_params=[(5, 5, 20),(5, 5, 20),(5, 5, 20),(5, 5, 20),(5, 5, 10), (5, 5, 10),(5, 5, 10)],
    likelihood_r_sched = [0.2, 0.15, 0.1, 0.04, 0.02 , 0.01, 0.001]
)

point_cloud_image = perception_state.process_depth_to_point_cloud_image(observation.depth)
segmentation_image  = perception_state.segment_scene(
    observation.rgb, point_cloud_image, observation.camera_pose, f"dashboard.png",
    FLOOR_THRESHOLD=0.01,
    TOO_CLOSE_THRESHOLD=0.2,
    FAR_AWAY_THRESHOLD=0.8,
)






occluder_image

pose ~ posePrior
id ~ idPrior
rendered_image = renderer(pose, id)
rendered_iamge_masked = apply_mask(rendered_image, occluder_image)
observed_image ~ 3dp3likelihood(rendered_iamge_masked)


observed_image = Condition on full observed depth image --- x
Use a segmentation mask to isolate a region of the depth image -- 
Apply occluder mask

