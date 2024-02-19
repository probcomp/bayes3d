import bayes3d as b
import jax.numpy as jnp
import jax
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import trimesh
from dcolmap.hgps.pose import Pose
import viser
server = viser.ViserServer()


intrinsics = b.Intrinsics(
    height=100,
    width=100,
    fx=200.0, fy=200.0,
    cx=50., cy=50.,
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


meshes = []

path = os.path.join(b.utils.get_assets_dir(), "sample_objs/bunny.obj")
bunny_mesh = trimesh.load(path)
bunny_mesh.vertices  = bunny_mesh.vertices * jnp.array([1.0, -1.0, 1.0]) + jnp.array([0.0, 1.0, 0.0])
meshes.append(bunny_mesh)


all_vertices = [jnp.array(mesh.vertices) for mesh in meshes]
all_faces = [jnp.array(mesh.faces) for mesh in meshes]
vertices_lens = jnp.array([len(verts) for verts in all_vertices])
vertices_lens_cumsum = jnp.pad(jnp.cumsum(vertices_lens),(1,0))
faces_lens = jnp.array([len(faces) for faces in all_faces])
faces_lens_cumsum = jnp.pad(jnp.cumsum(faces_lens),(1,0))

vertices = jnp.concatenate(all_vertices, axis=0)
vertices = jnp.concatenate([vertices, jnp.ones((vertices.shape[0], 1))], axis=-1)
faces = jnp.concatenate([faces + vertices_lens_cumsum[i] for (i,faces) in enumerate(all_faces)])

resolution = jnp.array([intrinsics.height, intrinsics.width])
