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


class MCS_Observation:
    def __init__(self, rgb, depth, intrinsics, segmentation, cam_pose):
        """RGBD Image
        
        Args:
            rgb (np.array): RGB image
            depth (np.array): Depth image
            camera_pose (np.array): Camera pose. 4x4 matrix
            intrinsics (b.camera.Intrinsics): Camera intrinsics
            segmentation (np.array): Segmentation image
        """
        self.rgb = rgb
        self.depth = depth
        self.intrinsics = intrinsics
        self.segmentation  = segmentation
        self.cam_pose = cam_pose

def get_obs_from_step_metadata(step_metadata, intrinsics, cam_pose):
    rgb = np.array(list(step_metadata.image_list)[-1])
    depth = np.array(list(step_metadata.depth_map_list)[-1])
    seg = np.array(list(step_metadata.object_mask_list)[-1])
    colors, seg_final_flat = np.unique(seg.reshape(-1,3), axis=0, return_inverse=True)
    seg_final = seg_final_flat.reshape(seg.shape[:2])
    observation = MCS_Observation(rgb, depth, intrinsics, seg_final, cam_pose)
    return observation

def cam_pose_from_step_metadata(step_metadata):
    cam_pose_diff_orientation = np.array([
        [ 1,0,0,0],
        [0,0,-1,-4.5], # 4.5 is an arbitrary value
        [ 0,1,0,step_metadata.camera_height],
        [ 0,0,0,1]
    ])
    inv_cam_pose = np.linalg.inv(cam_pose_diff_orientation)
    inv_cam_pose[1:3] *= -1
    cam_pose = np.linalg.inv(inv_cam_pose)
    return cam_pose

def intrinsics_from_step_metadata(step_metadata):
    width, height = step_metadata.camera_aspect_ratio
    aspect_ratio = width / height
    cx, cy = width / 2.0, height / 2.0
    fov_y = np.deg2rad(step_metadata.camera_field_of_view)
    fov_x = 2 * np.arctan(aspect_ratio * np.tan(fov_y / 2.0))
    fx = cx / np.tan(fov_x / 2.0)
    fy = cy / np.tan(fov_y / 2.0)
    clipping_near, clipping_far = step_metadata.camera_clipping_planes
    intrinsics = {
        'width' : width,
        'height' : height,
        'cx' : cx,
        'cy' : cy,
        'fx' : fx,
        'fy' : fy,
        'near' : clipping_near,
        'far' : clipping_far
    }
    return intrinsics

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
    
    with open(f"final_result_{scene_ID}.pkl", "rb") as file:
        final_result = pickle.load(file)

    print("The final rating for the {}th scene is ".format(i+1), final_result['rating'])

    controller.end_scene(*final_result.values())

    os.remove(f"{scene_ID}.npz")
    os.remove(f"final_result_{scene_ID}.pkl")