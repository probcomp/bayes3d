import jax.numpy as jnp
import jax



def render_planes_multiobject(poses, shape_planes, shape_dims, h,w, fx,fy, cx,cy):
    r, c = jnp.meshgrid(jnp.arange(w), jnp.arange(h))
    pixel_coords = jnp.stack([r,c],axis=-1)
    pixel_coords_dir = jnp.concatenate([(pixel_coords - jnp.array([cx,cy])) /  jnp.array([fx,fy]), jnp.ones((h,w,1))],axis=-1)

    return render_planes_multiobject_rays(poses, shape_planes, shape_dims, pixel_coords_dir)

def render_planes_multiobject_rays(poses, shape_planes, shape_dims, pixel_coords_dir):
    plane_poses = jnp.einsum("...ij,...ajk",poses, shape_planes).reshape(-1, 4, 4)
    shape_dimensions = shape_dims.reshape(-1, 2)


    denoms = jnp.einsum("ijk,ak->ija", pixel_coords_dir , plane_poses[:,:3,2])
    numerators = jnp.einsum("...k,...k", 
        plane_poses[:,:3,3],
        plane_poses[:,:3,2]
    )
    d = numerators / (denoms + 1e-10)
    points_temp = jnp.einsum("...ij,...kj", d[:,:,:,None], pixel_coords_dir[:,:,:,None])
    points = jnp.concatenate([points_temp, jnp.ones((*points_temp.shape[:3],1,))],axis=-1) # (H,W,N,4)
    inv_plane_poses = jnp.linalg.inv(plane_poses)
    points_in_plane_frame = jnp.einsum("...ij,ab...j->ab...i", inv_plane_poses, points)

    valid = jnp.all(jnp.abs(points_in_plane_frame[:,:,:,:2]) < shape_dimensions,axis=-1) * (points[:,:,:,2] > 0.0)# (H,W,N)
    z_vals = (20000.0 - points[:,:,:,2]) * valid
    idxs = jnp.argmax(z_vals, axis=-1)

    points_final = (
        points[jnp.arange(points.shape[0])[:, None], jnp.arange(points.shape[1])[None, :], idxs]
        *
        jnp.any(valid, axis=-1)[:,:,None]
    )

    return points_final


def get_rectangular_prism_shape(dimensions):
    half_width = dimensions / 2.0
    cube_plane_poses = jnp.array(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, half_width[2]],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, -half_width[2]],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, half_width[1]],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, -half_width[1]],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.0, 0.0, 1.0, half_width[0]],
                [0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            [
                [0.0, 0.0, 1.0, -half_width[0]],
                [0.0, 1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ]
    )

    plane_dimensions = jnp.array(
        [[half_width[0], half_width[1]], [half_width[0], half_width[1]], [half_width[0], half_width[2]], [half_width[0], half_width[2]], [half_width[2], half_width[1]], [half_width[2], half_width[1]]]
    )
    return cube_plane_poses, plane_dimensions



def batch_split(proposals, num_batches):
    num_proposals = proposals.shape[0]
    if num_proposals % num_batches != 0:
        # print(f"WARNING: {num_proposals} Not evenly divisible by {num_batches}; defaulting to 3x split")  # TODO find a good factor
        num_batches = 3
    return jnp.array(jnp.split(proposals, num_batches))

# Run the `scorer_parallel` scorer over the batched proposals `proposals_batches`, on `gt_image`
def batched_scorer_parallel_params(scorer_parallel, num_batches, proposals, parameters):
    def _score_batch(carry, proposals):  
        # score over the selected rotation proposals
        weights_new = scorer_parallel(proposals, parameters)
        return 0, weights_new  # return highest weight pose proposal encountered so far
    proposals_batches = batch_split(proposals, num_batches)
    _, batched_weights = jax.lax.scan(_score_batch, 0, proposals_batches)

    return batched_weights.ravel()

def batched_scorer_parallel(scorer_parallel, num_batches, proposals):
    def _score_batch(carry, proposals):  
        # score over the selected rotation proposals
        weights_new = scorer_parallel(proposals)
        return 0, weights_new  # return highest weight pose proposal encountered so far
    proposals_batches = batch_split(proposals, num_batches)
    _, batched_weights = jax.lax.scan(_score_batch, 0, proposals_batches)

    return batched_weights.ravel()