import jax.numpy as jnp
from  .transforms_3d import transform_from_axis_angle, transform_from_pos


def get_contact_planes(dimensions):
    return jnp.stack([
        transform_from_pos(jnp.array([0.0, 0.0, dimensions[2]/2.0])).dot(transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), 0.0)),
        transform_from_pos(jnp.array([0.0, 0.0, -dimensions[2]/2.0])).dot(transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.pi)),
        transform_from_pos(jnp.array([-dimensions[0]/2.0, 0.0, 0.0])).dot(transform_from_axis_angle(jnp.array([0.0, 1.0, 0.0]), -jnp.pi/2)),
        transform_from_pos(jnp.array([dimensions[0]/2.0, 0.0, 0.0])).dot(transform_from_axis_angle(jnp.array([0.0, 1.0, 0.0]), jnp.pi/2)),
        transform_from_pos(jnp.array([0.0, dimensions[1]/2.0, 0.0])).dot(transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), -jnp.pi/2)),
        transform_from_pos(jnp.array([0.0, -dimensions[1]/2.0, 0.0])).dot(transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.pi/2)),
    ])

def get_contact_transform(x,y,ang):
    return (
        transform_from_pos(jnp.array([x,y, 0.0])).dot(
            transform_from_axis_angle(jnp.array([1.0, 1.0, 0.0]), jnp.pi).dot(
                transform_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), ang)
            )
        )
    )