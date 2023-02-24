import jax3dp3 as j
import numpy as np
import subprocess
import h5py
import os

def render_blenderproc(mesh_paths, poses, camera_pose, intrinsics, scaling_factor=1.0):
    K = j.camera.K_from_intrinsics(intrinsics)
    np.savez("/tmp/blenderproc.npz", 
        mesh_paths=mesh_paths,
        scaling_factor=scaling_factor,
        poses=poses,
        camera_pose=camera_pose,
        K=K,
        height=intrinsics.height,
        width=intrinsics.width
    )

    bproc_script_path = os.path.join(os.path.dirname(j.__file__), "blenderproc/blenderproc_exec.py")
    subprocess.run([f"blenderproc run {bproc_script_path} /tmp/blenderproc.npz"], shell=True)

    rgb = None
    depth = None
    with h5py.File("/tmp/0.hdf5", 'r') as data:
        keys = data.keys()
        print(keys)
        depth = np.array(data["depth"])
        rgb = np.array(data["colors"])
    return rgb, depth