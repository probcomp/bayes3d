import jax.numpy as jnp
from .transforms_3d import apply_transform

def render_planes_multiobject_augmented(poses, shape_planes, shape_dims, h,w, fx_fy, cx_cy):
    plane_poses = jnp.einsum("...ij,...ajk",poses, shape_planes)
    num_planes_per_object = shape_planes.shape[1]
    shape_dimensions = shape_dims

    r, c = jnp.meshgrid(jnp.arange(w), jnp.arange(h))
    pixel_coords = jnp.stack([r,c],axis=-1)
    pixel_coords_dir = jnp.concatenate([(pixel_coords - cx_cy) / fx_fy, jnp.ones((h,w,1))],axis=-1)

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
    print('inv_object_poses.shape:');print(inv_object_poses.shape)
    points_in_object_frame = jnp.einsum("...ij,ab...kj->ab...ki", inv_object_poses, points)
    print('points_in_object_frame.shape:');print(points_in_object_frame.shape)


    valid = jnp.all(jnp.abs(points_in_plane_frame[:,:,:,:,:2]) < shape_dimensions,axis=-1) # (H,W,N, M)
    print('points.shape:');print(points.shape)
    print('valid.shape:');print(valid.shape)
    intersection_points = points * valid[:,:,:,:,None]
    print('intersection_points.shape:');print(intersection_points.shape)
    idxs_pre = jnp.argmax(intersection_points[:,:,:,:,2], axis=-1)
    print('idxs_pre.shape:');print(idxs_pre.shape)
    intersection_points_2 = intersection_points[
        jnp.arange(intersection_points.shape[0])[:, None,None],
        jnp.arange(intersection_points.shape[1])[None,:, None],
        jnp.arange(intersection_points.shape[2])[None,None,:],
        idxs_pre,
        :
    ]
    idxs = jnp.argmax(intersection_points_2[:,:,:,2], axis=-1)
    print('idxs.shape:');print(idxs.shape)

    collided = jnp.any(jnp.any(valid, axis=-1), axis=-1)
    print('collided.shape:');print(collided.shape)
    segmentation = (collided * idxs) + (1.0 - collided) * -1.0
    print('intersection_points_2.shape:');print(intersection_points_2.shape)
    points_final = (
        intersection_points_2[
            jnp.arange(intersection_points_2.shape[0])[:, None],
            jnp.arange(intersection_points_2.shape[1])[None, :],
            idxs,
        ]
        *
        collided[:,:,None]
    )

    return points_final, segmentation, points_in_object_frame