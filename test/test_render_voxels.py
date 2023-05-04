import numpy as np
import jax.numpy as jnp
import jax
import bayes3d as j
import trimesh
import os
import time


intrinsics = j.Intrinsics(
    300,
    300,
    200.0,200.0,
    150.0,150.0,
    0.001, 50.0
)
renderer = j.Renderer(intrinsics)

voxel_size = 0.05
voxel = j.mesh.make_cuboid_mesh(jnp.ones(3) * voxel_size)
renderer.add_mesh(voxel)

poses = jax.vmap(j.t3d.transform_from_pos)(jnp.array(np.random.rand(1000,3)))

start = time.time()
a = renderer.render_parallel(poses, 0)
print(time.time() - start)


start = time.time()
a = renderer.render_multiobject(poses, [0 for _ in range(len(poses))])
print(time.time() - start)