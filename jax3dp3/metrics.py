import jax.numpy as jnp
from jax3dp3.transforms_3d import rotation_matrix_to_quaternion
from jax3dp3.rendering import render_planes 

def get_rot_error_from_poses(p1, p2)->float:
    rot1 = p1[:3, :3] 
    rot2 = p2[:3, :3] 
    q1, q2 = rotation_matrix_to_quaternion(rot1), rotation_matrix_to_quaternion(rot2)

    rot_err = jnp.arccos(jnp.min(jnp.array([1.0, 2 * q1.dot(q2)**2 - 1]))) 

    return rot_err

def get_translation_error_from_poses(shape_renderer, p1, p2):
    coords1 = shape_renderer(p1)
    coords2 = shape_renderer(p2)

    c1_nonzero = coords1[coords1[:,:,2] != 0, :3]
    c2_nonzero = coords2[coords2[:,:,2] != 0, :3]
    
    if c1_nonzero.shape[0] == 0:  # not visible at all in cam frame
        c1_centroid = jnp.array([jnp.inf, jnp.inf, jnp.inf])
    else:
        c1_centroid = jnp.mean(c1_nonzero, axis=0)

    if c2_nonzero.shape[0] == 0:
        c2_centroid = jnp.array([-jnp.inf, -jnp.inf, -jnp.inf])
    else:
        c2_centroid = jnp.mean(c2_nonzero, axis=0)
    
    return sum(abs(jnp.abs(c1_centroid - c2_centroid)))

    # save_depth_image(best_img[:,:,2], f"c2f_out.png",max=max_depth)   