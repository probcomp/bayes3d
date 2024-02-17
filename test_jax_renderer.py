import bayes3d as b
import jax.numpy as jnp
import jax
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import trimesh

import viser
server = viser.ViserServer()

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

meshes = []


path = os.path.join(b.utils.get_assets_dir(), "sample_objs/cube.obj")
mesh = trimesh.load(path)
mesh.vertices  = mesh.vertices * jnp.array([1.0, 1.0, 1.0]) * 0.7
meshes.append(mesh)

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



import functools
@functools.partial(
    jnp.vectorize,
    signature="(2),(),(m,4,4),()->(3)",
    excluded=(
        4,
        5,
        6,
    ),
)
def interpolate_(uv, triangle_id, poses, object_id, vertices, faces, ranges):
    relevant_vertices = vertices[faces[triangle_id-1]]
    pose_of_object = poses[object_id-1]
    relevant_vertices_transformed = relevant_vertices @ pose_of_object.T
    barycentric = jnp.concatenate([uv, jnp.array([1.0 - uv.sum()])])
    interpolated_value = (relevant_vertices_transformed[:,:3] * barycentric.reshape(3,1)).sum(0)
    return interpolated_value


object_indices = jnp.array([0, 1])
ranges = jnp.hstack([faces_lens_cumsum[object_indices].reshape(-1,1), faces_lens[object_indices].reshape(-1,1)])


poses = jnp.array([b.transform_from_pos(jnp.array([0.0, 0.0, 5.0]))]*5)
poses = poses.at[:, 1,3].set(jnp.linspace(-0.2, 0.5, len(poses)))
poses2 = poses.at[:, 1,3].set(jnp.linspace(-0.0, 1.5, len(poses)))
poses2 = poses2.at[:, 0,3].set(-0.5)
poses = jnp.stack([poses, poses2], axis=1)

rast_out, rast_out_aux = jax_renderer.rasterize(
    poses,
    vertices,
    faces,
    ranges,
    projection_matrix,
    resolution
)
uvs = rast_out[...,:2]
object_ids = jnp.rint(rast_out_aux[...,0]).astype(jnp.int32)
triangle_ids = jnp.rint(rast_out_aux[...,1]).astype(jnp.int32)
mask = object_ids > 0

interpolated_values = interpolate_(uvs, triangle_ids, poses[:,None, None,:,:], object_ids, vertices, faces, ranges)
image = interpolated_values * mask[...,None]

server.reset_scene()
server.add_point_cloud(
    "image1",
    points=np.array(image[0]).reshape(-1,3),
    colors=np.array([1.0, 0.0, 0.0]),
    point_size=0.01
)
server.add_point_cloud(
    "image2",
    points=np.array(image[1]).reshape(-1,3),
    colors=np.array([0.0, 0.0, 0.0]),
    point_size=0.01
)

from IPython import embed; embed()