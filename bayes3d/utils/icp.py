import jax.numpy as jnp
import jax
import functools
import numpy as np
import bayes3d as b

import functools
@functools.partial(
    jnp.vectorize,
    signature='(m)->(z)',
    excluded=(1,2,3,),
)
def find_closest_point_at_pixel(
    ij,
    observed_xyz: jnp.ndarray,
    rendered_xyz_padded: jnp.ndarray,
    filter_size,
):
    rendered_filter = jax.lax.dynamic_slice(rendered_xyz_padded, (ij[0], ij[1], 0), (2*filter_size + 1, 2*filter_size + 1, 3))
    distances = jnp.linalg.norm(
        observed_xyz[ij[0], ij[1], :3] - rendered_filter,
        axis=-1
    )
    best_point = rendered_filter[jnp.unravel_index(jnp.argmin(distances), distances.shape)]
    return best_point


def get_nearest_neighbor(
    observed_xyz: jnp.ndarray,
    rendered_xyz: jnp.ndarray,
    filter_size: int
):
    rendered_xyz_padded = jax.lax.pad(rendered_xyz,  -100.0, ((filter_size,filter_size,0,),(filter_size,filter_size,0,),(0,0,0,)))
    jj, ii = jnp.meshgrid(jnp.arange(observed_xyz.shape[1]), jnp.arange(observed_xyz.shape[0]))
    indices = jnp.stack([ii,jj],axis=-1)
    matches = find_closest_point_at_pixel(
        indices,
        observed_xyz,
        rendered_xyz_padded,
        filter_size
    )
    return matches


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
    transform =  b.transform_from_rot_and_pos(rot_final, T)
    return transform

def icp_images(img, img_reference, init_pose, error_threshold, iterations, intrinsics, filter_size):
    def _icp_step(i, pose_and_error):
        pose, _ = pose_and_error
        transformed_cloud = b.apply_transform(img, pose)
        matches_in_img_reference = b.utils.get_nearest_neighbor(
            transformed_cloud,
            img_reference,
            filter_size
        )
        mask = (img[:,:,2] < intrinsics.far) *  (matches_in_img_reference[:,:,2] < intrinsics.far)
        avg_error = (jnp.linalg.norm(img - matches_in_img_reference,axis=-1) *mask).sum() / mask.sum()
        transform = b.utils.find_least_squares_transform_between_clouds(
            transformed_cloud.reshape(-1,3),
            matches_in_img_reference.reshape(-1,3),
            mask.reshape(-1,1)
        )
        return jnp.where(avg_error < error_threshold, pose, transform @ pose), avg_error
    return jax.lax.fori_loop(0, iterations, _icp_step, (init_pose,0.0))

