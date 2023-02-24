import jax3dp3 as j
import numpy as np
import subprocess
import h5py
import os


# 1. Adapt kubric_exec to consume the npz data that is written to disk by this function
# 2. This may require conveting poses (which are 4x4 transformation matrices) to positions and quaternions
# 3. Then adapt the subprocess call below to make the docker call and instead of calling examples/helloworld.py, it should be calling photorealistric_renderers/kubric_exec.py
# 4. In the final part of this function load the rgb and depth data that Kubric generated, and return it

def render_kubric(mesh_paths, poses, camera_pose, intrinsics, scaling_factor=1.0):
    K = j.camera.K_from_intrinsics(intrinsics)
    np.savez("/tmp/scene_data.npz", 
        mesh_paths=mesh_paths,
        scaling_factor=scaling_factor,
        poses=poses,
        camera_pose=camera_pose,
        K=K,
        height=intrinsics.height,
        width=intrinsics.width
    )

    kubric_script_path = os.path.join(os.path.dirname(j.__file__), "photorealistic_renderers/kubric_exec.py")
    subprocess.run(["DOCKERS STUFFF"], shell=True)

    #Load RGB and depth images from file
