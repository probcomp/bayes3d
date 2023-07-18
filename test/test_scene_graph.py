import bayes3d as b
import jax.numpy as jnp
import jax


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
