import jax.numpy as jnp
from jax3dp3.utils import extract_2d_patches
from jax3dp3.transforms_3d import (
    transform_from_rot_and_pos
)
import jax
import functools


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

def find_least_squares_transform_between_clouds(c1, c2, mask):
    centroid1 = jnp.sum(c1 * mask, axis=0) / jnp.sum(mask)
    centroid2 = jnp.sum(c2 * mask, axis=0) / jnp.sum(mask)
    c1_centered = c1 - centroid1
    c2_centered = c2 - centroid2
    H = jnp.transpose(c1_centered * mask).dot(c2_centered * mask)

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
