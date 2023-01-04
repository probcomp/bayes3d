from .transforms_3d import rotation_from_axis_angle, transform_from_rot_and_pos, transform_from_pos, transform_from_axis_angle
import jax.numpy as jnp
import networkx as nx
import jax

def get_contact_planes(dimensions):
    return jnp.stack(
        [
            transform_from_pos(jnp.array([0.0, dimensions[1]/2.0, 0.0])).dot(transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), -jnp.pi/2)),
            transform_from_pos(jnp.array([0.0, -dimensions[1]/2.0, 0.0])).dot(transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.pi/2)),
            transform_from_pos(jnp.array([0.0, 0.0, dimensions[2]/2.0])).dot(transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), 0.0)),
            transform_from_pos(jnp.array([0.0, 0.0, -dimensions[2]/2.0])).dot(transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.pi)),
            transform_from_pos(jnp.array([-dimensions[0]/2.0, 0.0, 0.0])).dot(transform_from_axis_angle(jnp.array([0.0, 1.0, 0.0]), -jnp.pi/2)),
            transform_from_pos(jnp.array([dimensions[0]/2.0, 0.0, 0.0])).dot(transform_from_axis_angle(jnp.array([0.0, 1.0, 0.0]), jnp.pi/2)),
        ]
    )

def get_contact_transform(contact_params):
    x,y,angle = contact_params
    return (
        transform_from_pos(jnp.array([x,y, 0.0])).dot(
            transform_from_axis_angle(jnp.array([1.0, 1.0, 0.0]), jnp.pi).dot(
                transform_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), angle)
            )
        )
    )

def relative_pose_from_contact(
    contact_params,
    face_params,
    dims_parent, dims_child,
):
    parent_plane = get_contact_planes(dims_parent)[face_params[0]]
    child_plane = get_contact_planes(dims_child)[face_params[1]]
    contact_transform = get_contact_transform(contact_params)
    return (parent_plane.dot(contact_transform)).dot(jnp.linalg.inv(child_plane))

def pose_from_contact(
    contact_params,
    face_params,
    dims_parent, dims_child,
    parent_pose
):
    parent_plane = get_contact_planes(dims_parent)[face_params[0]]
    child_plane = get_contact_planes(dims_child)[face_params[1]]
    contact_transform = get_contact_transform(contact_params)
    return parent_pose.dot(parent_plane.dot(contact_transform)).dot(jnp.linalg.inv(child_plane))

## Get poses


def iter(poses, box_dims, edge, contact_params, face_params):
    i, j = edge
    rel_pose = relative_pose_from_contact(contact_params, face_params, box_dims[i], box_dims[j])
    return (
        poses[i].dot(rel_pose) * (i != -1)
        +
        poses[j] * (i == -1)
    )

def absolute_poses_from_scene_graph(start_poses, box_dims, edges, contact_params, face_params):
    def _f(poses, _):
        new_poses = jax.vmap(iter, in_axes=(None, None, 0, 0, 0,))(poses, box_dims, edges, contact_params, face_params)
        return (new_poses, new_poses)
    return jax.lax.scan(_f, start_poses, jnp.ones(edges.shape[0]))[0]