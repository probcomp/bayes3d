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
    fx=75.0, fy=75.0,
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
faces = mesh.faces

vertices_h = jnp.hstack([vertices, jnp.ones((vertices.shape[0], 1))])

poses =jnp.array([jnp.eye(4)]*1000)

def xfm_points(points, matrix):
    points2 = jnp.concatenate([points, jnp.ones((*points.shape[:-1], 1))], axis=-1)
    return jnp.matmul(points2, matrix.T)

object_pose = b.transform_from_pos(jnp.array([0.0, 0.0, 3.0]))
final_mtx_proj = projection_matrix @ object_pose
posw = jnp.concatenate([vertices, jnp.ones((*vertices.shape[:-1], 1))], axis=-1)
pos_clip_ja = xfm_points(vertices, final_mtx_proj)
rast_out, rast_out_db = jax_renderer.rasterize(
    poses,
    pos_clip_ja,
    faces,
    jnp.array([intrinsics.height, intrinsics.width]),
)
img = rast_out[150,...,3]
b.get_depth_image(img).save("test.png")



shape_keep = gb_pos.shape

gb_pos, _ = jax_renderer.interpolate(
    posw[None, ...], rast_out, faces, rast_out_db, jnp.array([0, 1, 2, 3])
)
gb_pos = gb_pos[..., :3]
depth = xfm_points(gb_pos, object_pose)
depth = depth.reshape(shape_keep)[..., 2] * -1