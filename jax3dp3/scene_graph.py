import jax.numpy as jnp
import jax
import jax3dp3.enumerations
import jax3dp3.transforms_3d as t3d

def contact_planes(dimensions):
    return jnp.stack(
        [
            # bottom
            t3d.transform_from_pos(jnp.array([0.0, dimensions[1]/2.0, 0.0])).dot(t3d.transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), -jnp.pi/2)),
            # top
            t3d.transform_from_pos(jnp.array([0.0, -dimensions[1]/2.0, 0.0])).dot(t3d.transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.pi/2)),
            # back
            t3d.transform_from_pos(jnp.array([0.0, 0.0, dimensions[2]/2.0])).dot(t3d.transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), 0.0)),
            # front
            t3d.transform_from_pos(jnp.array([0.0, 0.0, -dimensions[2]/2.0])).dot(t3d.transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.pi)),
            # left
            t3d.transform_from_pos(jnp.array([-dimensions[0]/2.0, 0.0, 0.0])).dot(t3d.transform_from_axis_angle(jnp.array([0.0, 1.0, 0.0]), -jnp.pi/2)),
            # right
            t3d.transform_from_pos(jnp.array([dimensions[0]/2.0, 0.0, 0.0])).dot(t3d.transform_from_axis_angle(jnp.array([0.0, 1.0, 0.0]), jnp.pi/2)),
        ]
    )

def get_contact_transform(contact_params):
    x,y,angle = contact_params
    return (
        t3d.transform_from_pos(jnp.array([x,y, 0.0])).dot(
            t3d.transform_from_axis_angle(jnp.array([1.0, 1.0, 0.0]), jnp.pi).dot(
                t3d.transform_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), angle)
            )
        )
    )

def relative_pose_from_contact(
    contact_params,
    face_parent, face_child,
    dims_parent, dims_child,
):
    parent_plane = contact_planes(dims_parent)[face_parent]
    child_plane = contact_planes(dims_child)[face_child]
    contact_transform = get_contact_transform(contact_params)
    return (parent_plane.dot(contact_transform)).dot(jnp.linalg.inv(child_plane))

def pose_from_contact(
    contact_params,
    face_parent, face_child,
    dims_parent, dims_child,
    parent_pose
):
    return parent_pose.dot(relative_pose_from_contact(contact_params, face_parent, face_child, dims_parent, dims_child))

def pose_from_contact_and_face_params(
    contact_params,
    face_child,
    dims_child,
    contact_plane
):
    child_plane = contact_planes(dims_child)[face_child]
    contact_transform = get_contact_transform(contact_params)
    return (contact_plane.dot(contact_transform)).dot(jnp.linalg.inv(child_plane))


pose_from_contact_and_face_params_parallel_jit = jax.jit(jax.vmap(pose_from_contact_and_face_params, in_axes=(0, 0, None, None)))


def get_contact_plane(
    parent_pose,
    dims_parent,
    parent_face,
):
    parent_plane = contact_planes(dims_parent)[parent_face]
    return parent_pose.dot(parent_plane)

## Get poses


def iter(poses, box_dims, edge, contact_params, face_parent, face_child):
    i, j = edge
    rel_pose = relative_pose_from_contact(contact_params, face_parent, face_child, box_dims[i], box_dims[j])
    return (
        poses[i].dot(rel_pose) * (i != -1)
        +
        poses[j] * (i == -1)
    )

def absolute_poses_from_scene_graph(start_poses, box_dims, edges, contact_params, face_parent, face_child):
    def _f(poses, _):
        new_poses = jax.vmap(iter, in_axes=(None, None, 0, 0, 0, 0))(poses, box_dims, edges, contact_params, face_parent, face_child)
        return (new_poses, new_poses)
    return jax.lax.scan(_f, start_poses, jnp.ones(edges.shape[0]))[0]
absolute_poses_from_scene_graph_jit = jax.jit(absolute_poses_from_scene_graph)

def enumerate_contact_and_face_parameters(min_x,min_y,min_angle, max_x, max_y, max_angle, num_x, num_y, num_angle, faces):
    contact_params_sweep = jax3dp3.enumerations.make_translation_grid_enumeration_3d(
        min_x,min_y, min_angle,
        max_x, max_y, max_angle,
        num_x, num_y, num_angle
    )
    contact_params_sweep_extended = jnp.tile(contact_params_sweep, (faces.shape[0],1))
    face_params_sweep = jnp.repeat(faces, contact_params_sweep.shape[0])
    return contact_params_sweep_extended, face_params_sweep
    
