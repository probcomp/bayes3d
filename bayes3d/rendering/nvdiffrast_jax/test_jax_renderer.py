import bayes3d as b
import jax.numpy as jnp
import jax
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import trimesh

intrinsics = b.Intrinsics(
    height=100,
    width=100,
    fx=200.0, fy=200.0,
    cx=50.0, cy=50.0,
    near=0.001, far=16.0
)

projection_matrix = b.camera._open_gl_projection_matrix(
    intrinsics.height,
    intrinsics.width,
    intrinsics.fx,
    intrinsics.fy,
    intrinsics.cx,
    intrinsics.cy,
    intrinsics.near,
    intrinsics.far,
)

from bayes3d.rendering.nvdiffrast_jax.jax_renderer import Renderer as JaxRenderer
jax_renderer = JaxRenderer(intrinsics)

path = os.path.join(b.utils.get_assets_dir(), "sample_objs/bunny.obj")
mesh  =trimesh.load(path)
mesh.vertices  = mesh.vertices * jnp.array([1.0, -1.0, 1.0]) + jnp.array([0.0, 1.0, 0.0])
vertices = mesh.vertices
vertices = jnp.concatenate([vertices, jnp.ones((vertices.shape[0], 1))], axis=-1)
faces = mesh.faces

poses =jnp.array([b.transform_from_pos(jnp.array([0.0, 0.0, 5.0]))]*1000)
poses = poses.at[:, 1,3].set(jnp.linspace(-0.4, 0.4, len(poses)))
rast_out, rast_out_db = jax_renderer.rasterize(
    poses,
    vertices,
    faces,
    projection_matrix,
    jnp.array([intrinsics.height, intrinsics.width]),
)
b.hstack_images(
    [
        b.get_depth_image((rast_out[i,...,3]) *1.0, remove_max=False)
        for i in [1, 500, 999]
    ]
).save("test2.png")


from IPython import embed; embed()

T = 0
points_transformed = b.apply_transform(vertices[:,:3], poses[T])

import viser
server = viser.ViserServer()
server.add_point_cloud("bunny", points=np.array(points_transformed)[:,:3], colors=np.zeros_like(points_transformed)[:,:3])
