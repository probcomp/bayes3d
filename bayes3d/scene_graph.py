import jax.numpy as jnp
import jax
import bayes3d.enumerations
import bayes3d.transforms_3d as t3d

def get_contact_planes(dimensions):
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

def relative_pose_from_edge(
    contact_params,
    face_child, dims_child,
):
    x,y,angle = contact_params
    contact_transform = (
        t3d.transform_from_pos(jnp.array([x,y, 0.0])).dot(
            t3d.transform_from_axis_angle(jnp.array([1.0, 1.0, 0.0]), jnp.pi).dot(
                t3d.transform_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), angle)
            )
        )
    )
    child_plane = get_contact_planes(dims_child)[face_child]
    return contact_transform.dot(jnp.linalg.inv(child_plane))

relative_pose_from_edge_jit = jax.jit(relative_pose_from_edge)
relative_pose_from_edge_parallel_jit = jax.jit(
    jax.vmap(
        relative_pose_from_edge,
        in_axes=(0, 0, 0),
    )
)

def iter(poses, box_dims, parent, child, contact_params, face_parent, face_child):
    parent_plane = get_contact_planes(box_dims[parent])[face_parent]
    relative = parent_plane.dot(
        relative_pose_from_edge(contact_params, face_child, box_dims[child])
    )
    return (
        poses[parent].dot(relative) * (parent != -1)
        +
        poses[child] * (parent == -1)
    )

def poses_from_scene_graph(start_poses, box_dims, parents, contact_params, face_parent, face_child):
    def _f(poses, _):
        new_poses = jax.vmap(iter, in_axes=(None, None, 0, 0, 0, 0, 0))(poses, box_dims, parents, jnp.arange(parents.shape[0]), contact_params, face_parent, face_child)
        return (new_poses, new_poses)
    return jax.lax.scan(_f, start_poses, jnp.ones(parents.shape[0]))[0]
poses_from_scene_graph_jit = jax.jit(poses_from_scene_graph)

    
