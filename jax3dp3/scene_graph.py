from .transforms_3d import rotation_from_axis_angle, transform_from_rot_and_pos, transform_from_pos, transform_from_axis_angle
import jax.numpy as jnp
import networkx as nx

def get_contact_planes(dimensions):
    return jnp.stack([
        transform_from_pos(jnp.array([0.0, dimensions[1]/2.0, 0.0])).dot(transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), -jnp.pi/2)),
        transform_from_pos(jnp.array([0.0, -dimensions[1]/2.0, 0.0])).dot(transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.pi/2)),
        transform_from_pos(jnp.array([0.0, 0.0, dimensions[2]/2.0])).dot(transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), 0.0)),
        transform_from_pos(jnp.array([0.0, 0.0, -dimensions[2]/2.0])).dot(transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.pi)),
        transform_from_pos(jnp.array([-dimensions[0]/2.0, 0.0, 0.0])).dot(transform_from_axis_angle(jnp.array([0.0, 1.0, 0.0]), -jnp.pi/2)),
        transform_from_pos(jnp.array([dimensions[0]/2.0, 0.0, 0.0])).dot(transform_from_axis_angle(jnp.array([0.0, 1.0, 0.0]), jnp.pi/2)),
    ])

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
    dims_parent, dims_child,
    parent_face_id, child_face_id,
    contact_params
):
    parent_plane = get_contact_planes(dims_parent)[parent_face_id]
    child_plane = get_contact_planes(dims_child)[child_face_id]
    contact_transform = get_contact_transform(contact_params)
    return (parent_plane.dot(contact_transform)).dot(jnp.linalg.inv(child_plane))


