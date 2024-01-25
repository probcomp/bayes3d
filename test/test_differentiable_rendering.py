import os

import jax
import jax.numpy as jnp
import numpy as np

import bayes3d as b
from bayes3d.rendering.nvdiffrast_jax.jax_renderer import Renderer as JaxRenderer

intrinsics = b.Intrinsics(
    height=200, width=200, fx=200.0, fy=200.0, cx=100.0, cy=100.0, near=0.01, far=5.5
)
jax_renderer = JaxRenderer(intrinsics)
# ---------------------
# Load object
# ---------------------
model_dir = os.path.join(b.utils.get_assets_dir(), "bop/ycbv/models")
idx = 14
mesh_path = os.path.join(model_dir, "obj_" + "{}".format(idx).rjust(6, "0") + ".ply")
m = b.utils.load_mesh(mesh_path)
m = b.utils.scale_mesh(m, 1.0 / 100.0)

vtx_pos = jnp.array(m.vertices.astype(np.float32))
pos_idx = jnp.array(m.faces.astype(np.int32))
print("Mesh has %d triangles and %d vertices." % (pos_idx.shape[0], vtx_pos.shape[0]))
resolution = jnp.array([200, 200])
pos = vtx_pos[None, ...]
pos = jnp.concatenate([pos, jnp.ones((*pos.shape[:-1], 1))], axis=-1)


def func(i):
    rast_out, rast_out_db = jax_renderer.rasterize(
        pos + i, pos_idx, jnp.array([200, 200])
    )
    return rast_out.mean()


func_jit = jax.jit(func)
print(func_jit(0.0))
grad_func = jax.value_and_grad(func)
val, grad = grad_func(0.0)
print(grad)

# # Test Torch
# import nvdiffrast.torch as dr   # modified nvdiffrast to expose backward fn call api
# torch_glctx = dr.RasterizeGLContext()
# device = torch.device("cuda")
# def func_torch(i):
#     rast_out, rast_out_db  = dr.rasterize(torch_glctx, torch.tensor(np.array(pos),device=device) + i, torch.tensor(np.array(pos_idx),device=device), resolution=torch.tensor(np.array(resolution),device=device))
#     return rast_out.mean()
# input_vec = torch.tensor([0.0],  device=device, requires_grad=True)
# loss = func_torch(input_vec)
# print(loss)
# loss.backward()
# print(input_vec.grad)


def func(i):
    rast_out, rast_out_db = jax_renderer.rasterize(
        pos + i, pos_idx, jnp.array([200, 200])
    )
    colors, _ = jax_renderer.interpolate(
        pos + i, rast_out, pos_idx, rast_out_db, jnp.array([0, 1, 2, 3])
    )
    return colors.mean()


func_jit = jax.jit(func)
print(func_jit(0.0))
grad_func = jax.value_and_grad(func)
val, grad = grad_func(0.0)
print(grad)

# # Test Torch
# import nvdiffrast.torch as dr   # modified nvdiffrast to expose backward fn call api
# torch_glctx = dr.RasterizeGLContext()
# device = torch.device("cuda")
# def func_torch(i):
#     rast_out, rast_out_db  = dr.rasterize(torch_glctx, torch.tensor(np.array(pos),device=device) + i, torch.tensor(np.array(pos_idx),device=device), resolution=torch.tensor(np.array(resolution),device=device))
#     colors,_ = dr.interpolate( torch.tensor(np.array(pos),device=device) + i, rast_out, torch.tensor(np.array(pos_idx),device=device), rast_out_db)
#     return colors.mean()
# input_vec = torch.tensor([0.0],  device=device, requires_grad=True)
# loss = func_torch(input_vec)
# print(loss)
# loss.backward()
# print(input_vec.grad)
