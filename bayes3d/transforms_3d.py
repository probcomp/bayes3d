import jax
import jax.numpy as jnp
import numpy as np
from typing import Tuple
import cv2


def inverse_pose(t):
    return jnp.linalg.inv(t)

def transform_from_pos(t):
    return jnp.vstack(
        [jnp.hstack([jnp.eye(3), t.reshape(3,1)]), jnp.array([0.0, 0.0, 0.0, 1.0])]
    )

def transform_from_rot(rot):
    return jnp.vstack(
        [jnp.hstack([rot, jnp.zeros((3,1))]), jnp.array([0.0, 0.0, 0.0, 1.0])]
    )

def transform_from_rot_and_pos(rot, t):
    return jnp.vstack(
        [jnp.hstack([rot, t.reshape(3,1)]), jnp.array([0.0, 0.0, 0.0, 1.0])]
    )

def rotation_from_axis_angle(axis, angle):
    sina = jnp.sin(angle)
    cosa = jnp.cos(angle)
    direction = axis / jnp.linalg.norm(axis)
    # rotation matrix around unit vector
    R = jnp.diag(jnp.array([cosa, cosa, cosa]))
    R = R + jnp.outer(direction, direction) * (1.0 - cosa)
    direction = direction * sina
    R = R + jnp.array([[0.0, -direction[2], direction[1]],
                        [direction[2], 0.0, -direction[0]],
                        [-direction[1], direction[0], 0.0]])
    return R

def rotation_from_rodrigues(r_in):
    r_flat = r_in.reshape(-1)
    theta = jnp.linalg.norm(r_flat)
    r = r_flat/theta
    A = jnp.array([[0, -r[2], r[1]],[r[2], 0, -r[0]],[-r[1], r[0],  0]])
    R = jnp.cos(theta) * jnp.eye(3) + (1 - jnp.cos(theta)) * r.reshape(-1,1) * r.transpose() + jnp.sin(theta) * A
    return jnp.where(theta < 0.0001, jnp.eye(3), R)

def transform_to_posevec(transform):
    rvec = jnp.array(cv2.Rodrigues(np.array(transform[:3,:3]))[0]).reshape(-1)
    tvec = transform[:3,3].reshape(-1)
    posevec = jnp.concatenate([tvec, rvec])
    return posevec

def transform_from_posevec(posevec):
    return transform_from_rot_and_pos(rotation_from_rodrigues(posevec[3:]), posevec[:3])

def axis_angle_from_rotation(R):
    rvec = rodrigues_from_rotation(R)
    return rvec/jnp.linalg.norm(rvec), jnp.linalg.norm(rvec)

def rodrigues_from_rotation(R):
    #formula from http://motion.pratt.duke.edu/RoboticSystems/3DRotations.html
    eps = 1e-8 #prevent numerical instability
    theta = jnp.clip(jnp.arccos((jnp.trace(R)-1)/2), a_min=eps, a_max=jnp.pi-eps)
    rvec = jnp.array([(R[2,1]-R[1,2])/(2*jnp.sin(theta)), (R[0,2]-R[2,0])/(2*jnp.sin(theta)), (R[1,0]-R[0,1])/(2*jnp.sin(theta))]) * theta
    rvec += eps
    return rvec.reshape(-1)

def axis_angle_from_rotation(R):
    rvec = rodrigues_from_rotation(R)
    return rvec/jnp.linalg.norm(rvec), jnp.linalg.norm(rvec)

def rodrigues_from_rotation(R):
    #formula from http://motion.pratt.duke.edu/RoboticSystems/3DRotations.html
    eps = 1e-9 #prevent numerical instability, results in mismatch from OpenCV implementation at singularity point theta=0
    theta = jnp.clip(jnp.arccos((jnp.trace(R)-1)/2), a_min=eps, a_max=jnp.pi-eps)
    rvec = jnp.array([(R[2,1]-R[1,2])/(2*jnp.sin(theta)), (R[0,2]-R[2,0])/(2*jnp.sin(theta)), (R[1,0]-R[0,1])/(2*jnp.sin(theta))]) * theta
    rvec += eps
    return rvec.reshape(-1)

def transform_to_posevec_j(R):
    rot_vec = rodrigues_from_rotation(R[:3,:3])
    t_vec = R[:3,3].flatten()
    return jnp.hstack((t_vec, rot_vec))

def transform_from_rvec_tvec(rvec, tvec):
    return transform_from_rot_and_pos(
        rotation_from_rodrigues(rvec),
        tvec.reshape(-1)
    )

def identity_pose():
    return jnp.eye(4)

def transform_from_axis_angle(axis, angle):
    M = jnp.identity(4)
    M = M.at[:3, :3].set(rotation_from_axis_angle(axis, angle))
    return M

def add_homogenous_ones(cloud):
    return jnp.concatenate([cloud, jnp.ones((*cloud.shape[:-1],1))],axis=-1)

# move point cloud to a specified pose
# coords: (N,3) point cloud
# pose: (4,4) pose matrix. rotation matrix in top left (3,3) and translation in (:3,3)
def apply_transform(coords, transform):
    coords = jnp.einsum(
        'ij,...j->...i',
        transform,
        jnp.concatenate([coords, jnp.ones(coords.shape[:-1] + (1,))], axis=-1),
    )[..., :-1]
    return coords

apply_transform_jit = jax.jit(apply_transform)

def quaternion_to_rotation_matrix(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (q0,q1,q2,q3) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[0]
    q1 = Q[1]
    q2 = Q[2]
    q3 = Q[3]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = jnp.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix

def rotation_matrix_to_xyzw(matrix):
    wxyz = rotation_matrix_to_quaternion(matrix)
    return jnp.array([*wxyz[1:], wxyz[0]])

def xyzw_to_rotation_matrix(xyzw):
    return quaternion_to_rotation_matrix(jnp.array([xyzw[-1], *xyzw[:-1]]))

def pybullet_pose_to_transform(pybullet_pose):
    translation = jnp.array(pybullet_pose[0])
    R = xyzw_to_rotation_matrix(pybullet_pose[1])
    cam_pose = (
        transform_from_rot_and_pos(R, translation)
    )
    return cam_pose

def transform_to_pybullet_pose(pose):
    translation = jnp.array(pose[:3,3])
    quat = rotation_matrix_to_xyzw(pose[:3,:3])
    return translation, quat

def rotation_matrix_to_quaternion(matrix):

    def case0(m):
        t = 1 + m[0, 0] - m[1, 1] - m[2, 2]
        q = jnp.array(
            [
                m[2, 1] - m[1, 2],
                t,
                m[1, 0] + m[0, 1],
                m[0, 2] + m[2, 0],
            ]
        )
        return t, q

    def case1(m):
        t = 1 - m[0, 0] + m[1, 1] - m[2, 2]
        q = jnp.array(
            [
                m[0, 2] - m[2, 0],
                m[1, 0] + m[0, 1],
                t,
                m[2, 1] + m[1, 2],
            ]
        )
        return t, q

    def case2(m):
        t = 1 - m[0, 0] - m[1, 1] + m[2, 2]
        q = jnp.array(
            [
                m[1, 0] - m[0, 1],
                m[0, 2] + m[2, 0],
                m[2, 1] + m[1, 2],
                t,
            ]
        )
        return t, q

    def case3(m):
        t = 1 + m[0, 0] + m[1, 1] + m[2, 2]
        q = jnp.array(
            [
                t,
                m[2, 1] - m[1, 2],
                m[0, 2] - m[2, 0],
                m[1, 0] - m[0, 1],
            ]
        )
        return t, q

    # Compute four cases, then pick the most precise one.
    # Probably worth revisiting this!
    case0_t, case0_q = case0(matrix)
    case1_t, case1_q = case1(matrix)
    case2_t, case2_q = case2(matrix)
    case3_t, case3_q = case3(matrix)

    cond0 = matrix[2, 2] < 0
    cond1 = matrix[0, 0] > matrix[1, 1]
    cond2 = matrix[0, 0] < -matrix[1, 1]

    t = jnp.where(
        cond0,
        jnp.where(cond1, case0_t, case1_t),
        jnp.where(cond2, case2_t, case3_t),
    )
    q = jnp.where(
        cond0,
        jnp.where(cond1, case0_q, case1_q),
        jnp.where(cond2, case2_q, case3_q),
    )
    return q * 0.5 / jnp.sqrt(t)

def unproject_depth(depth, intrinsics):
    mask = (depth < intrinsics.far) * (depth > intrinsics.near)
    depth = depth * mask + intrinsics.far * (1.0 - mask)
    y, x = jnp.mgrid[: depth.shape[0], : depth.shape[1]]
    x = (x - intrinsics.cx) / intrinsics.fx
    y = (y - intrinsics.cy) / intrinsics.fy
    point_cloud_image = jnp.stack([x, y, jnp.ones_like(x)], axis=-1) * depth[:, :, None]
    return point_cloud_image

unproject_depth_jit = jax.jit(unproject_depth)
unproject_depth_vmap_jit = jax.jit(jax.vmap(unproject_depth, in_axes=(0,None)))

def transform_from_pos_target_up(pos, target, up):
    z = target- pos
    z = z / jnp.linalg.norm(z)

    x = jnp.cross(z, up)
    x = x / jnp.linalg.norm(x)

    y = jnp.cross(z,x)
    y = y / jnp.linalg.norm(y)

    R = jnp.hstack([
        x.reshape(-1,1),y.reshape(-1,1),z.reshape(-1,1)
    ])
    return transform_from_rot_and_pos(R, pos)

def estimate_transform_between_clouds(c1, c2):
    centroid1 = jnp.sum(c1, axis=0) 
    centroid2 = jnp.sum(c2, axis=0)
    c1_centered = c1 - centroid1
    c2_centered = c2 - centroid2
    H = jnp.transpose(c1_centered).dot(c2_centered)

    U,_,V = jnp.linalg.svd(H)
    rot = (jnp.transpose(V).dot(jnp.transpose(U)))

    modifier = jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, -1.0],
    ])
    V_mod = modifier.dot(V)
    rot2 = (jnp.transpose(V_mod).dot(jnp.transpose(U)))

    rot_final = (jnp.linalg.det(rot) < 0) * rot2 + (jnp.linalg.det(rot) > 0) * rot

    T = (centroid2 - rot_final.dot(centroid1))
    transform =  transform_from_rot_and_pos(rot_final, T)
    return transform

def interpolate_between_two_poses(pose_start, pose_end, time):
    """
    Interpolates between pose_start and pose_end.
            Parameters:
                    pose_start (jnp.ndarray): Start pose, 4x4 transformation matrix
                    pose_end (jnp.ndarray): End pose, 4x4 transformation matrix
                    time (float): Time between 0.0 and 1.0
            Returns:
                    pose_interpolated (jnp.ndarray): Interpolated pose, 4x4 transformation matrix
    """
    # [
    #     R t
    #     0 1
    # ]
    # TODO: Implement this function.
    # TODO: Unit test that shows pose_start, pose_end, and the end pose.
    # See test/test_meshcat.py for how to visualize a pose using b.show_pose.
    return pose_start

def sparsify_point_cloud(point_cloud, min_distance):
    """
    Select a subset of points from a point cloud that are at least min_distance apart.
            Parameters:
                    point_cloud (jnp.ndarray): 3D point cloud, Nx3.
                    min_distance (float): Desired minimum distance between points.
            Returns:
                    subsampled_point_cloud (jnp.ndarray)
    """
    # TODO: Implement this function.
    # TODO: Unit test that shows the orginal and sparsified clouds.
    # See test/test_meshcat.py for how to visualize point clouds using b.show_cloud.
    return point_cloud

def perspective_n_point(point_cloud, pixel_coordinates):
    """
        Given 3D coordinates of points and the locations of the points (pixel coordinates)
        in the image, estimate the pose of the camera.
            Parameters:
                    point_cloud (jnp.ndarray): 3D point cloud, Nx3.
                    pixel_coordinates (float): Pixel coordinates of the points in the image, Nx2.
            Returns:
                    camera_pose (jnp.ndarray): Estimated camera pose
    """
    # TODO: Implement this function.
    # TODO: Unit test that shows the correct pose being inferred.
    return jnp.eye(4)

