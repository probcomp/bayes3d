import jax.numpy as jnp
from bayes3d.utils.utils import extract_2d_patches
from bayes3d.transforms_3d import (
    transform_from_rot_and_pos
)
import jax
import functools
import numpy as np

def get_nearest_neighbor(
    obs_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
):
    rendered_xyz_patches = extract_2d_patches(rendered_xyz, (10,10))
    matches = find_closest_point_at_pixel(
        obs_xyz,
        rendered_xyz_patches,
    )
    return matches

@functools.partial(
    jnp.vectorize,
    signature='(m),(h,w,m)->(m)',
)
def find_closest_point_at_pixel(
    data_xyz: jnp.ndarray,
    model_xyz: jnp.ndarray,
):
    distance = jnp.linalg.norm(data_xyz - model_xyz, axis=-1)
    best_point = model_xyz[jnp.unravel_index(jnp.argmin(distance), distance.shape)]
    return best_point


def icp(render_func, init_pose, obs_img, outer_iterations, inner_iterations):
    def _icp_step(j, pose_):
        rendered_img = render_func(pose_)
        def _icp_step_inner(i, pose):
            neighbors = get_nearest_neighbor(obs_img, rendered_img)
            mask = (neighbors[:,:,2] > 0)  * (obs_img[:,:,2] > 0)
            c1 = neighbors[:,:,:3].reshape(-1,3)
            c2 = obs_img[:,:,:3].reshape(-1,3)

            transform = find_least_squares_transform_between_clouds(c1, c2, mask.reshape(-1,1))
            pose = transform.dot(pose)
            return pose
        return jax.lax.fori_loop(0, inner_iterations, _icp_step_inner, pose_)
    return jax.lax.fori_loop(0, outer_iterations, _icp_step, init_pose)


# Get transform to apply to a to make it close to b
def icp_open3d(a,b):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(a)

    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(b)
            
    reg_p2p = o3d.pipelines.registration.registration_icp(
        pcd, pcd2, 0.05, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    return reg_p2p.transformation