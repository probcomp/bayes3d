import jax
import jax.numpy as jnp
import numpy as np
from jax_renderer import Renderer
import os
import torch
import bayes3d as b


intrinsics = b.Intrinsics(
    height=200,
    width=200,
    fx=200.0, fy=200.0,
    cx=100.0, cy=100.0,
    near=0.01, far=5.5
)

r = Renderer(intrinsics)

model_dir = os.path.join(b.utils.get_assets_dir(),"bop/ycbv/models")
idx = 14
mesh_path = os.path.join(model_dir,"obj_" + "{}".format(idx).rjust(6, '0') + ".ply")
m = b.utils.load_mesh(mesh_path)
m = b.utils.scale_mesh(m, 1.0/100.0)

vtx_pos = jnp.asarray(m.vertices.astype(np.float32))
pos_idx = jnp.asarray(m.faces.astype(np.int32))
col_idx = jnp.asarray(np.zeros((vtx_pos.shape[0],3)).astype(np.int32))
vtx_col = jnp.asarray(np.ones((1,3)).astype(np.float32))
# print("Mesh has %d triangles and %d vertices." % (pos_idx.shape[0], pos.shape[0]))
print(vtx_pos.shape, pos_idx.shape)
resolution = (200, 200)

ret = r.rasterize(vtx_pos[None, :], pos_idx, resolution)

from IPython import embed; embed()
