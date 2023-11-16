from collections import namedtuple
import jax
import jax.numpy as jnp
import numpy as np
import os, argparse
import time
import torch
import bayes3d as b
from bayes3d.rendering.nvdiffrast_jax.jax_renderer import Renderer as JaxRenderer



intrinsics = b.Intrinsics(
    height=200,
    width=200,
    fx=200.0, fy=200.0,
    cx=100.0, cy=100.0,
    near=0.01, far=5.5
)
jax_renderer = JaxRenderer(intrinsics)
#---------------------
# Load object
#---------------------
model_dir = os.path.join(b.utils.get_assets_dir(),"bop/ycbv/models")
idx = 14
mesh_path = os.path.join(model_dir,"obj_" + "{}".format(idx).rjust(6, '0') + ".ply")
m = b.utils.load_mesh(mesh_path)
m = b.utils.scale_mesh(m, 1.0/100.0)

vtx_pos = jnp.array(m.vertices.astype(np.float32))
pos_idx = jnp.array(m.faces.astype(np.int32))
print("Mesh has %d triangles and %d vertices." % (pos_idx.shape[0], vtx_pos.shape[0]))

pos = vtx_pos[None,...]
pos = jnp.concatenate([pos, jnp.ones((*pos.shape[:-1],1))], axis=-1)
rast_out, rast_out_db = jax_renderer.rasterize(pos, pos_idx, jnp.array([200,200]))

def func(i):
    rast_out, rast_out_db = jax_renderer.rasterize(pos + i, pos_idx, jnp.array([200,200]))
    return rast_out.mean()

func_jit = jax.jit(func)
func_jit(0.0)

grad_func = jax.value_and_grad(func)
val,grad = grad_func(0.0)


