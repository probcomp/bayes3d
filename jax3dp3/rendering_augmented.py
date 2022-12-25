import jax.numpy as jnp
from .transforms_3d import apply_transform

def render_planes_multiobject_augmented(poses, shape_planes, shape_dims, h,w, fx,fy, cx,cy, offsets, scales):
    plane_poses = jnp.einsum("...ij,...ajk",poses, shape_planes)
    num_planes_per_object = shape_planes.shape[1]
    shape_dimensions = shape_dims

    r, c = jnp.meshgrid(jnp.arange(w), jnp.arange(h))
    pixel_coords = jnp.stack([r,c],axis=-1)
    pixel_coords_dir = jnp.concatenate([(pixel_coords - jnp.array([cx,cy])) / jnp.array([fx,fy]), jnp.ones((h,w,1))],axis=-1)

    denoms = jnp.einsum("ijk,abk->ijab", pixel_coords_dir , plane_poses[:,:,:3,2])
    numerators = jnp.einsum("...k,...k", 
        plane_poses[:,:,:3,3],
        plane_poses[:,:,:3,2]
    )
    d = numerators / (denoms + 1e-10)
    points_temp = jnp.einsum("...aij,...kj->...aik", d[:,:,:,:,None], pixel_coords_dir[:,:,:,None])
    points = jnp.concatenate([points_temp, jnp.ones((*points_temp.shape[:4],1,))],axis=-1) # (H,W,N,M,4)
    inv_plane_poses = jnp.linalg.inv(plane_poses)
    points_in_plane_frame = jnp.einsum("...ij,ab...j->ab...i", inv_plane_poses, points)

    inv_object_poses = jnp.linalg.inv(poses)
    points_in_object_frame = jnp.einsum("...ij,ab...kj->ab...ki", inv_object_poses, points)


    valid = jnp.all(jnp.abs(points_in_plane_frame[:,:,:,:,:2]) < shape_dimensions,axis=-1) # (H,W,N, M)
    intersection_points = points * valid[:,:,:,:,None]
    idxs_pre = jnp.argmax(intersection_points[:,:,:,:,2], axis=-1)
    intersection_points_2 = intersection_points[
        jnp.arange(intersection_points.shape[0])[:, None,None],
        jnp.arange(intersection_points.shape[1])[None,:, None],
        jnp.arange(intersection_points.shape[2])[None,None,:],
        idxs_pre,
        :
    ]
    points_in_object_frame = points_in_object_frame[
        jnp.arange(points_in_object_frame.shape[0])[:, None,None],
        jnp.arange(points_in_object_frame.shape[1])[None,:, None],
        jnp.arange(points_in_object_frame.shape[2])[None,None,:],
        idxs_pre,
        :
    ]
    idxs = jnp.argmax(intersection_points_2[:,:,:,2], axis=-1)

    collided = jnp.any(jnp.any(valid, axis=-1), axis=-1)
    segmentation = (collided * idxs) + (1 - collided) * -1
    points_final = (
        intersection_points_2[
            jnp.arange(intersection_points_2.shape[0])[:, None],
            jnp.arange(intersection_points_2.shape[1])[None, :],
            idxs,
        ]
        *
        collided[:,:,None]
    )

    points_in_object_frame = ((points_in_object_frame[
            jnp.arange(points_in_object_frame.shape[0])[:, None],
            jnp.arange(points_in_object_frame.shape[1])[None, :],
            idxs,
        ][...,:3] - offsets[idxs])/ (scales[idxs][...,None])
        *
        collided[:,:,None]
    )
    return points_final, segmentation, points_in_object_frame