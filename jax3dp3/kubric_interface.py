import jax3dp3 as j
import numpy as np
import jax.numpy as jnp
import subprocess
import os


# 1. Adapt kubric_exec to consume the npz data that is written to disk by this function
# 2. This may require conveting poses (which are 4x4 transformation matrices) to positions and quaternions
# 3. Then adapt the subprocess call below to make the docker call and instead of calling examples/helloworld.py, it should be calling photorealistric_renderers/kubric_exec.py
# 4. In the final part of this function load the rgb and depth data that Kubric generated, and return it

def render_multiobject_parallel(mesh_paths, poses, intrinsics, scaling_factor=1.0, lighting=5.0):
    #asset intrinsics are compatible with Blender camera parameters
    assert intrinsics.fx == intrinsics.fy, "fx and fy must be equal"
    assert intrinsics.cx == intrinsics.width/2, "cx must be width/2"
    assert intrinsics.cy == intrinsics.height/2, "cy must be height/2"

    K = j.camera.K_from_intrinsics(intrinsics)
    poses_pos_quat_all = []
    for ii in range(poses.shape[1]):
        poses_pos_quat = []
        for jj in range(poses.shape[0]):
            poses_pos_quat.append((
                np.array(poses[jj, ii,:3,3]),
                np.array(j.t3d.rotation_matrix_to_quaternion(poses[jj,ii,:3,:3]))
            ))
        poses_pos_quat_all.append(poses_pos_quat)

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
    
    command_string2 = f""" kubricdockerhub/kubruntu /usr/bin/python3 {path}/jax3dp3/photorealistic_renderers/kubric_exec_parallel.py"""
    print(command_string + command_strings + command_string2)
    subprocess.run([command_string + command_strings + command_string2], shell=True)

    rgbd_images = []
    for i in range(poses.shape[1]):
        data = np.load(f"/tmp/{i}.npz")
        rgb, seg, depth = data["rgba"], data["segmentation"][...,0], data["depth"][:,:,0]
        depth[depth > intrinsics.far] = intrinsics.far
        depth[depth < intrinsics.near] = intrinsics.near
        rgbd_images.append(j.RGBD(rgb,depth, camera_pose, intrinsics, seg))

    return rgbd_images