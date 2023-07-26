import bayes3d as j
import numpy as np
import jax.numpy as jnp
import subprocess
import os

def render_many(mesh_paths, poses, intrinsics, scaling_factor=1.0, lighting=50.0, camera_pose=None):
    """Render a scene with multiple objects in it through kubric.

    Args:
        mesh_paths (list): List of paths to meshes.
        poses (jnp.ndarray): Array of poses of shape (num_frames, num_objects, 4, 4).
        intrinsics (b.camera.Intrinsics): Camera intrinsics.
    Returns:
        list: List of rendered RGBD images.    
    """

    # warn if intrinsics are incompatible with Blender camera parameters 
    if intrinsics.fx != intrinsics.fy:
        print("fx is not equal to fy!")
    if intrinsics.cy != intrinsics.height/2:
        print("cy is not equal to height/2!")
    if intrinsics.cy != intrinsics.height/2:
        print("cy is not equal to height/2!")

    K = j.camera.K_from_intrinsics(intrinsics)
    poses_pos_quat_all = []
    for scene_index in range(poses.shape[0]):
        poses_pos_quat = []
        for object_index in range(poses.shape[1]):
            poses_pos_quat.append((
                np.array(poses[scene_index, object_index,:3,3]),
                np.array(j.t3d.rotation_matrix_to_quaternion(poses[scene_index,object_index,:3,:3]))
            ))
        poses_pos_quat_all.append(poses_pos_quat)

    if camera_pose is None:
        camera_pose = jnp.eye(4)
        camera_pose = camera_pose @ j.t3d.transform_from_axis_angle(jnp.array([1.0, 0.0,0.0]), jnp.pi)
    cam_pose_pos_quat = (np.array(camera_pose[:3,3]), np.array(j.t3d.rotation_matrix_to_quaternion(camera_pose[:3,:3])))


    np.savez("/tmp/blenderproc_kubric.npz", 
        mesh_paths=mesh_paths,
        scaling_factor=scaling_factor,
        poses=np.array(poses_pos_quat_all, dtype=object) ,
        camera_pose=np.array(cam_pose_pos_quat, dtype=object),
        K=K,
        height=intrinsics.height,
        width=intrinsics.width,
        fx = intrinsics.fx,
        fy = intrinsics.fy,
        cx = intrinsics.cx,
        cy = intrinsics.cy,
        near = intrinsics.near,
        far = intrinsics.far,
        intensity=lighting
    )

    path = os.path.dirname(os.path.dirname(__file__))
    print('path:');print(path)

    command_string = f"""sudo docker run --rm --interactive --user $(id -u):$(id -g) --volume {path}:{path} --volume /tmp:/tmp  """
    command_strings = "".join([
        f""" --volume {os.path.dirname(p)}:{os.path.dirname(p)} """ for p in mesh_paths
    ])
    command_string2 = f""" kubricdockerhub/kubruntu /usr/bin/python3 {path}/photorealistic_renderers/_kubric_exec_parallel.py"""
    print(command_string + command_strings + command_string2)
    subprocess.run([command_string + command_strings + command_string2], shell=True)

    rgbd_images = []
    for i in range(poses.shape[0]):
        data = np.load(f"/tmp/{i}.npz")
        rgb, seg, depth = data["rgba"], data["segmentation"][...,0], data["depth"][:,:,0]
        depth[depth > intrinsics.far] = intrinsics.far
        depth[depth < intrinsics.near] = intrinsics.near
        rgbd_images.append(j.RGBD(rgb,depth, camera_pose, intrinsics, seg))

    return rgbd_images