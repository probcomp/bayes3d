import numpy as np
import jax.numpy as jnp
import jax
import bayes3d as b
import trimesh
import os
import time


intrinsics = b.Intrinsics(
    300,
    300,
    200.0,200.0,
    150.0,150.0,
    0.001, 50.0
)
b.setup_renderer(intrinsics)


b.RENDERER.add_mesh(b.mesh.make_cuboid_mesh(jnp.array([0.1, 0.1, 0.1])))
b.RENDERER.add_mesh(b.mesh.make_cuboid_mesh(jnp.array([0.2, 0.2, 0.2])))


obsevation = b.RENDERER.render_multiobject(jnp.tile(jnp.eye(4)[None, ...], (2, 1, 1)), jnp.array([0,1]))[:,:,:3]

poses = jnp.tile(jnp.eye(4)[None, None, ...], (2, 500, 1, 1))
traces = b.Traces(
    poses, jnp.array([0,1]), jnp.array([0.1, 0.2,0.3]), jnp.array([0.01, 0.02]),
    1000.0, obsevation
)

scores = b.score_traces_jit(traces)

from IPython import embed; embed()