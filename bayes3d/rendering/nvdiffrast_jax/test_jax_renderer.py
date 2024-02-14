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

path = os.path.join(b.utils.get_assets_dir(), "sample_objs/bunny.obj")
mesh  =trimesh.load(path)
mesh.vertices  = mesh.vertices * jnp.array([1.0, -1.0, 1.0]) + jnp.array([0.0, 1.0, 0.0])
vertices = mesh.vertices
vertices = jnp.concatenate([vertices, jnp.ones((vertices.shape[0], 1))], axis=-1)
faces = jnp.array(mesh.faces)

poses =jnp.array([b.transform_from_pos(jnp.array([0.0, 0.0, 5.0]))]*1000)
poses = poses.at[:, 1,3].set(jnp.linspace(-1.4, 0.2, len(poses)))

parallel_render, _ = jax_renderer.rasterize(
    poses,
    vertices,
    faces,
    projection_matrix,
    jnp.array([intrinsics.height, intrinsics.width]),
)

images = []
for i in [0, int(len(poses)/2), len(poses)-1]:
    images.append(b.get_depth_image((parallel_render[i,...,3]) *1.0, remove_max=False))
b.hstack_images(
    images
).save("sweep.png")

test_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
for i in test_indices:
    individual, rast_out_db = jax_renderer.rasterize(
        poses[i:i+1],
        vertices,
        faces,
        projection_matrix,
        jnp.array([intrinsics.height, intrinsics.width]),
    )
    assert jnp.allclose(parallel_render[i], individual[0])





# uvs = rast_out[...,:2]
# triangle_ids = rast_out[...,3:4].astype(jnp.int32)

# mask = rast_out[...,2] > 0
# b.get_depth_image((mask[0]) *1.0, remove_max=False).save("mask.png")

# import functools
# @functools.partial(
#     jnp.vectorize,
#     signature="(2),(1),(4,4)->(3)",
#     excluded=(
#         3,
#         4,
#     ),
# )
# def interpolate_(uv, triangle_id, pose, vertices, faces):
#     u,v = uv
#     relevant_vertices = vertices[faces[triangle_id-1][0]]
#     relevant_vertices_transformed = relevant_vertices @ pose.T
#     barycentric = jnp.concatenate([uv, jnp.array([1.0 - uv.sum()])])
#     interpolated_value = (relevant_vertices_transformed[:,:3] * barycentric.reshape(3,1)).sum(0)
#     return interpolated_value

# interpolated_values = interpolate_(uvs, triangle_ids, poses[...,None, None,:,:], vertices, faces)
# image = interpolated_values * mask[...,None]



# T = 0
# points_transformed = b.apply_transform(vertices[:,:3], poses[T])


# server.add_point_cloud(
#     "bunny",
#     points=np.array(points_transformed)[:,:3],
#     colors=np.zeros_like(points_transformed)[:,:3]
# )

# server.add_point_cloud(
#     "image",
#     points=np.array(image[T]).reshape(-1,3),
#     colors=np.array([1.0, 0.0, 0.0])
# )


from IPython import embed; embed()
