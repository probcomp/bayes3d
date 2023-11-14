from collections import namedtuple
import jax
import jax.numpy as jnp
import numpy as np
import os, sys
import time
import torch
import bayes3d as b
from jax_renderer import Renderer as JaxRenderer

JAX_RENDERER = os.environ['JAX']=="True" if 'JAX' in os.environ else True

#--------------------
# Setup 
#--------------------
intrinsics = b.Intrinsics(
    height=200,
    width=200,
    fx=200.0, fy=200.0,
    cx=100.0, cy=100.0,
    near=0.01, far=5.5
)
proj_cam = torch.from_numpy(np.array(b.camera._open_gl_projection_matrix(
    intrinsics.height, intrinsics.width, 
    intrinsics.fx, intrinsics.fy, 
    intrinsics.cx, intrinsics.cy, 
    intrinsics.near, intrinsics.far
))).cuda()


#---------------------
# Load object
#---------------------
model_dir = os.path.join(b.utils.get_assets_dir(),"bop/ycbv/models")
idx = 14
mesh_path = os.path.join(model_dir,"obj_" + "{}".format(idx).rjust(6, '0') + ".ply")
m = b.utils.load_mesh(mesh_path)
m = b.utils.scale_mesh(m, 1.0/100.0)

vtx_pos = torch.from_numpy(m.vertices.astype(np.float32)).cuda()
pos_idx = torch.from_numpy(m.faces.astype(np.int32)).cuda()
col_idx = torch.from_numpy(np.zeros((vtx_pos.shape[0],3)).astype(np.int32)).cuda()
vtx_col = torch.from_numpy(np.ones((1,3)).astype(np.float32)).cuda()
# print("Mesh has %d triangles and %d vertices." % (pos_idx.shape[0], pos.shape[0]))
print(vtx_pos.shape, pos_idx.shape)

#----------------------
# Get clip-space poses (torch)
#----------------------
rot_mtx_44 = torch.tensor([[-0.9513,  0.1699,  0.2573,  0.0000],
                            [-0.2436,  0.0976, -0.9650,  0.0000],
                            [-0.1890, -0.9806, -0.0514,  2.5000],
                            [ 0.0000,  0.0000,  0.0000,  1.0000]],).cuda()
pos = vtx_pos[None,...]
# posw = torch.cat([pos, torch.ones([pos.shape[0], pos.shape[1], 1]).cuda()], axis=2)  # (xyz) -> (xyz1)
transform_mtx = torch.matmul(proj_cam, rot_mtx_44)  # transform = projection + pose rotation
pos_clip_ja = torch.matmul(
    torch.nn.functional.pad(pos, pad=(0, 1), mode="constant", value=1.0),
    torch.transpose(transform_mtx[None,...], 1, 2),
)

pos_clip_ja_jax = jnp.array(pos_clip_ja.cpu())
pos_idx_jax = jnp.array(pos_idx.cpu())
resolution = jnp.array([200,200])
rast_out_shp = (len(pos_clip_ja_jax), resolution[0], resolution[1], 4)

# pos_clip_ja = dd.xfm_points(pos, transform_mtx[None,...])  # transform points
#----------------------
# Rasterize
#----------------------
KEY1, KEY2 = jax.random.split(jax.random.PRNGKey(0), num=2)

# Test 1. JAX rasterize fwd.  Check that JAX and Torch rasterize fwd match
# Test 2

dy, ddb = torch.ones(rast_out_shp).cuda(), torch.ones(rast_out_shp).cuda()
dy_jax, ddb_jax = jnp.ones(rast_out_shp), jnp.ones(rast_out_shp)



ranges = torch.empty(size=(0, 2), dtype=torch.int32, device='cpu')

from IPython import embed; embed()

JAX_RENDERER = True

if JAX_RENDERER:
    r = JaxRenderer(intrinsics)
    from jax_renderer import _rasterize_bwd_custom_call, _rasterize_fwd_custom_call
else:
    import nvdiffrast.torch as dr
    glctx = dr.RasterizeGLContext()

# Test 1
print(f"Test 2 JAX_RENDERRER={JAX_RENDERER}")
if JAX_RENDERER:
    output_jax, _ = _rasterize_fwd_custom_call(r, jnp.array(pos_clip_ja_jax), jnp.array(pos_idx_jax), resolution)
    print(output_jax[output_jax > 0])
    print(jnp.isnan(output_jax).sum())
    print(jnp.where(output_jax > 0))
    gradients = _rasterize_bwd_custom_call(r, jnp.array(pos_clip_ja_jax), jnp.array(pos_idx_jax), jnp.array(output_jax), jnp.array(dy_jax), jnp.array(ddb_jax))[0]
    print(gradients)
    print(jnp.isnan(gradients).sum())
    print(gradients.sum())
else:
    output_torch, _ =  dr._get_plugin(gl=True).rasterize_fwd_gl(glctx.cpp_wrapper, pos_clip_ja, pos_idx, resolution, ranges, -1)
    print(output_torch[output_torch > 0])
    grad_torch = dr._get_plugin().rasterize_grad_db( pos_clip_ja, pos_idx, output_torch, dy, ddb)
    print(grad_torch.sum())




# Test 1
print(f"Test 1 JAX_RENDERRER={JAX_RENDERER}")
if JAX_RENDERER:
    output_jax, _= r.rasterize(pos_clip_ja_jax, pos_idx_jax, resolution)
    print(output_jax[output_jax > 0])
else:
    output_torch, _ = dr.rasterize(glctx, pos_clip_ja, pos_idx, resolution=resolution)
    print(output_torch[output_torch > 0])


# if JAX_RENDERER:

#     # jit with dummy inputs
#     (_o, _db), _rast_vjp = jax.vjp(r.rasterize, 
#                                 jnp.zeros_like(pos_clip_ja_jax), 
#                                 jnp.ones_like(pos_idx_jax), 
#                                 resolution)
#     _rast_vjp((jnp.zeros_like(_o) + 1e-12, jnp.zeros_like(_db) + 1e-12))

#     dummy_dy = jax.random.uniform(KEY1, rast_out_shp) 
#     dummy_ddb = jax.random.uniform(KEY2, rast_out_shp) 

#     # evaluate and time.
#     start_time = time.time()
#     (rast_out, rast_out_db), rasterize_vjp = jax.vjp(r.rasterize, 
#                                                      pos_clip_ja_jax, 
#                                                      pos_idx_jax, 
#                                                      resolution)
#     pos_grads = rasterize_vjp((dummy_dy, 
#                                dummy_ddb))[0]
#     print("JAX BWD RESULTS=", pos_grads.sum(), pos_grads.min(), pos_grads.max())  # TODO this returns nans
#     end_time = time.time()
#     print(f"JAX rasterization (eval + grad): {(end_time - start_time)*1000} ms")

#     b.viz.get_depth_image(rast_out[0][:,:,2]).save("img_jax.png")
# else:
#     resolution = [200,200]
#     TorchCtx = namedtuple('TorchCtx', ['saved_tensors', 'saved_grad_db'])
#     rast_out_shp = (len(pos_clip_ja), resolution[0], resolution[1], 4)

#     dummy_dy = torch.from_numpy(np.asarray(jax.random.uniform(KEY1, rast_out_shp))).cuda()
#     dummy_ddb = torch.from_numpy(np.asarray(jax.random.uniform(KEY2, rast_out_shp))).cuda()

#     # evaluate and time.
#     start_time = time.time()
#     rast_out, rast_out_db  = dr.rasterize(glctx, pos_clip_ja, pos_idx, resolution=resolution)
#     ctx = TorchCtx(saved_tensors=(pos_clip_ja, pos_idx, rast_out), saved_grad_db=True)
#     pos_grads = dr._rasterize_func.backward(ctx, dummy_dy, dummy_ddb)[1]   # 7 outputs; all are None except pos input
#     print("TORCH BWD RESULTS=", pos_grads.sum(), pos_grads.min(), pos_grads.max())
#     end_time = time.time()
    
#     print(f"Torch rasterization (eval + grad): {(end_time - start_time)*1000} ms")

#     b.viz.get_depth_image(jnp.array(rast_out[0][:,:,2].cpu())).save("img_torch.png")


# #----------------------
# # Interpolate
# #----------------------
# if JAX_RENDERER:
#     posw_jax = jnp.array(posw.cpu())

#     # jit with dummy input
#     r.interpolate(jnp.zeros_like(posw_jax), jnp.zeros_like(rast_out), jnp.ones_like(pos_idx_jax))

#     start_time = time.time()
#     gb_pos, _dummy = r.interpolate(posw_jax, rast_out, pos_idx_jax)
#     end_time = time.time()
    
#     print(f"JAX interpolation: {(end_time - start_time)*1000} ms")
#     b.viz.get_depth_image(gb_pos[0][:,:,2]).save("interpolate_jax.png")
# else:
#     start_time = time.time()
#     gb_pos, _dummy = dr.interpolate(posw, rast_out, pos_idx)
#     end_time = time.time()
    
#     print(f"JAX interpolation: {(end_time - start_time)*1000} ms")
#     b.viz.get_depth_image(jnp.array(gb_pos[0][:,:,2].cpu())).save("interpolate_torch.png")


from IPython import embed; embed()


