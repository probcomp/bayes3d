import bayes3d as b
import bayes3d.genjax
import jax
import os
import jax.numpy as jnp
import genjax

key = jax.random.PRNGKey(1)

b.setup_visualizer()


intrinsics = b.Intrinsics(
    height=100,
    width=100,
    fx=300.0, fy=300.0,
    cx=50.0, cy=50.0,
    near=0.01, far=20.0
)

b.setup_renderer(intrinsics)
b.RENDERER.add_mesh_from_file(os.path.join(b.utils.get_assets_dir(), "sample_objs/cube.obj"))
b.RENDERER.add_mesh_from_file(os.path.join(b.utils.get_assets_dir(), "sample_objs/cube.obj"))

importance_jit = jax.jit(b.genjax.model.importance)

key, (_,gt_trace) = importance_jit(key, genjax.choice_map({
    "parent_0": -1,
    "camera_pose": jnp.eye(4),
    "id_0": jnp.int32(0),
}), (jnp.arange(1), jnp.arange(2), jnp.array(jnp.ones(3)*5.0), jnp.array([0.2, 0.2, -2*jnp.pi]), b.RENDERER.model_box_dims, 100.0))
b.genjax.viz_trace_meshcat(gt_trace)
