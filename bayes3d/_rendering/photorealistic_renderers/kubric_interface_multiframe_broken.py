import bayes3d as j
import numpy as np
import jax.numpy as jnp
import subprocess
import os


# 1. Adapt kubric_exec to consume the npz data that is written to disk by this function
# 2. This may require conveting poses (which are 4x4 transformation matrices) to positions and quaternions
# 3. Then adapt the subprocess call below to make the docker call and instead of calling examples/helloworld.py, it should be calling photorealistric_renderers/kubric_exec.py
# 4. In the final part of this function load the rgb and depth data that Kubric generated, and return it

def render_many(mesh_paths, object_poses, dome_pose, cam_poses, intrinsics, frames, seed=0, scaling_factor=1.0, lighting=5.0):
    """Render a scene with multiple objects in it through kubric.

    Args:
        mesh_paths (list): List of paths to meshes.
        poses (jnp.ndarray): Array of poses of shape (num_objects, 4, 4).
        intrinsics (b.camera.Intrinsics): Camera intrinsics.
    Returns:
        list: List of rendered RGBD images.    
    """

    #asset intrinsics are compatible with Blender camera parameters
    assert intrinsics.fx == intrinsics.fy, "fx and fy must be equal"
    assert intrinsics.cx == intrinsics.width/2, "cx must be width/2"
    assert intrinsics.cy == intrinsics.height/2, "cy must be height/2"

    K = j.camera.K_from_intrinsics(intrinsics)

    #set up object quaternions
    obj_poses_pos_quat = []
    for object_index in range(object_poses.shape[0]):
        obj_poses_pos_quat.append((
            np.array(object_poses[object_index,:3,3]),
            np.array(j.t3d.rotation_matrix_to_quaternion(object_poses[object_index,:3,:3]))
        ))

    #set up camera quaternions
    cam_poses_pos_quat = []
    for cam_index in range(cam_poses.shape[0]):
        cam_poses_pos_quat.append((
            np.array(cam_poses[cam_index,:3,3]),
            np.array(j.t3d.rotation_matrix_to_quaternion(cam_poses[cam_index,:3,:3]))
        ))

    #set up dome quaternion
    dome_pose_pos_quat = (np.array(dome_pose[:3,3]), np.array(j.t3d.rotation_matrix_to_quaternion(dome_pose[:3,:3])))

    np.savez("/tmp/blenderproc_kubric.npz", 
        mesh_paths=mesh_paths,
        scaling_factor=scaling_factor,
        obj_poses=np.array(obj_poses_pos_quat, dtype=object) ,
        camera_poses=np.array(cam_poses_pos_quat, dtype=object),
        K=K,
        height=intrinsics.height,
        width=intrinsics.width,
        fx = intrinsics.fx,
        fy = intrinsics.fy,
        cx = intrinsics.cx,
        cy = intrinsics.cy,
        near = intrinsics.near,
        far = intrinsics.far,
        intensity=lighting,
        dome_pose=np.array(dome_pose_pos_quat, dtype=object),
        frames=frames,
        seed=seed
    )

    path = os.path.dirname(os.path.dirname(__file__))
    print('path:');print(path)

    command_string = f"""sudo docker run --rm --interactive --user $(id -u):$(id -g) --volume {path}:{path} --volume /tmp:/tmp  """
    command_strings = "".join([
        f""" --volume {os.path.dirname(p)}:{os.path.dirname(p)} """ for p in mesh_paths
    ])
    
    command_string2 = f""" kubricdockerhub/kubruntu /usr/bin/python3 {path}/photorealistic_renderers/_kubric_exec_parallel.py"""

    command_string3 = f" --hdri_assets={path}/assets/HDRI_haven/HDRI_haven.json"

    command_string4 = f" --kubasic_assets={path}/assets/KuBasic/KuBasic.json"

    #print(command_string + command_strings + command_string3 + command_string4)
    subprocess.run([command_string + command_strings + command_string2 + command_string3 + command_string4], shell=True)

    rgbd_images = []

    cam_positions = []
    cam_orientations = []

    for i in range(frames): 
        data = np.load(f"/tmp/{i}.npz")
        rgb, seg, depth, position, orientation = data["rgba"], data["segmentation"][...,0], data["depth"][:,:,0], data["position"], data["orientation"]
        depth[depth > intrinsics.far] = intrinsics.far
        depth[depth < intrinsics.near] = intrinsics.near
        rgbd_images.append(j.RGBD(rgb,depth, cam_poses[i], intrinsics, seg))
        cam_positions.append(position)
        cam_orientations.append(orientation)

    return rgbd_images, np.array(cam_positions), np.array(cam_orientations)