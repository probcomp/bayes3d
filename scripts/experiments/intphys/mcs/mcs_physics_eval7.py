import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'False'
import bayes3d as b
from tqdm import tqdm
import numpy as np
import zmq
import pickle5
import pickle
import subprocess
import zlib
import sys
import machine_common_sense as mcs
import glob
from mcs_utils import *

scene_folder = sys.argv[1]

files = sorted(glob.glob(scene_folder + "/*.json"))
controller = mcs.create_controller("mcs/config_level2.ini")

# NOTE: REMINDER TO PUT THIS WHOILE THING INTO A TRY EXCCEPT
for i, file in enumerate(files):
    print(f"Scene Count: {i+1}/{len(files)}")
    scene_data = mcs.load_scene_json_file(file)
    step_metadata = controller.start_scene(scene_data)
    scene_intrinsics = intrinsics_from_step_metadata(step_metadata)
    scene_cam_pose = cam_pose_from_step_metadata(step_metadata)
    MCS_Observations = [get_obs_from_step_metadata(step_metadata, scene_intrinsics, scene_cam_pose)]

    def MCS_stepper():
        while True:
            yield

    for _ in tqdm(MCS_stepper()):
        step_metadata = controller.step("Pass")
        if len(step_metadata.action_list) == 0:
            break
        MCS_Observations.append(get_obs_from_step_metadata(step_metadata, scene_intrinsics, scene_cam_pose))  # Do stuff here

    print("Observations loaded")
    scene_ID = np.random.randint(79384572398)
    print(f"Scene ID randomly generated as {scene_ID}")
    np.savez(f"{scene_ID}.npz", MCS_Observations)
    command = ["python", "unified_mcs_physics_eval7.py", str(scene_ID)]

    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
        # The "stdout" attribute of the result object contains the command's standard output
        print("Command Output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        # If the command returns a non-zero exit code, it will raise a CalledProcessError
        print(f"Command failed with exit code {e.returncode}")
        print("Error Output:")
        print(e.stderr)
    
    try:
        with open(f"final_result_{scene_ID}.pkl", "rb") as file:
            final_result = pickle.load(file)
    except:
        print("SCENE FAILED --> RANDOM CHOICE ACTIVATED")
        rand_ = np.random.choice([0,1])
        final_result = {'rating':rand_, 'score':float(rand_)}

    print("The final rating for the {}th scene is ".format(i+1), final_result['rating'])

    controller.end_scene(*final_result.values())

    os.remove(f"{scene_ID}.npz")
    os.remove(f"final_result_{scene_ID}.pkl")