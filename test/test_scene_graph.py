import bayes3d as b
import jax.numpy as jnp
import jax
import os


N = 4
scene_graph = b.scene_graph.SceneGraph(
    root_poses= jnp.tile(jnp.eye(4)[None,...],(N,1,1)),
    box_dimensions = jnp.ones((N,3)),
    parents = jnp.array([-1, 0, 0, 2]),
    contact_params = jax.random.uniform(jax.random.PRNGKey(10),(N,3), minval=-1.0, maxval=1.0),
    face_parent = jnp.array([0 ,1, 1, 2]),
    face_child = jnp.array([2 ,3, 4, 5])
)
scene_graph.visualize("graph.png", node_names=["table", "apple", "can", "banana"])

floating_scene_graph = scene_graph.create_floating_scene_graph()
assert jnp.isclose(floating_scene_graph.get_poses(), scene_graph.get_poses()).all()


parent_object_index = 0
child_object_index = 1
parent_pose = scene_graph.get_poses()[parent_object_index]
child_pose = scene_graph.get_poses()[child_object_index]
face_parent = scene_graph.face_parent[child_object_index]
face_child = scene_graph.face_child[child_object_index]
dims_parent = scene_graph.box_dimensions[parent_object_index]
dims_child = scene_graph.box_dimensions[child_object_index]

parent_contact_plane = parent_pose @ b.scene_graph.get_contact_planes(dims_parent)[face_parent]
child_contact_plane = child_pose @ b.scene_graph.get_contact_planes(dims_child)[face_child]

contact_params, slack = b.scene_graph.closest_approximate_contact_params(parent_contact_plane, child_contact_plane)
assert jnp.isclose(slack[:3,3], 0.0, atol=1e-7).all()
assert jnp.isclose(slack[:3,:3], jnp.eye(3), atol=1e-7).all()

assert jnp.isclose(contact_params, scene_graph.contact_params[child_object_index]).all()

from IPython import embed; embed()
