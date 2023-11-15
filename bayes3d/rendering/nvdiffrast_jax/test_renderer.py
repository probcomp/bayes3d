from collections import namedtuple
import jax
import jax.numpy as jnp
import numpy as np
import os, argparse
import time
import torch
import bayes3d as b
from jax_renderer import Renderer as JaxRenderer
# import nvdiffrast.torch as dr
import bayes3d.rendering.nvdiffrast_full.nvdiffrast.torch as dr   # modified nvdiffrast to expose backward fn call api


# --------------
# Which renderer to test
# --------------
parser=argparse.ArgumentParser()
parser.add_argument('TEST_NAME', type=str, help="jax or torch", default="jax")
args = parser.parse_args()

JAX_RENDERER = args.TEST_NAME == 'jax'
print(f"Testing JAX: {JAX_RENDERER}")

#--------------------
# Setup renderers 
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

if JAX_RENDERER:
    # setup Jax renderer
    jax_renderer = JaxRenderer(intrinsics)
else:
    # setup Torch renderer
    torch_glctx = dr.RasterizeGLContext()


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
print("Mesh has %d triangles and %d vertices." % (pos_idx.shape[0], vtx_pos.shape[0]))


#--------------------
# transform points op
#--------------------
def xfm_points(points, matrix):
    """Transform points.
    Args:
        points: Tensor containing 3D points with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
        use_python: Use PyTorch's torch.matmul (for validation)
    Returns:
        Transformed points in homogeneous 4D with shape [minibatch_size, num_vertices, 4].
    """
    out = torch.matmul(
        torch.nn.functional.pad(points, pad=(0, 1), mode="constant", value=1.0),
        torch.transpose(matrix, 1, 2),
    )
    return out

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
pos_clip_ja = xfm_points(pos, transform_mtx[None,...])  # transform points

resolution = [200,200]
rast_out_shp = (len(pos_clip_ja), resolution[0], resolution[1], 4)


#---------------------
# Test setup
#---------------------
# randomly create manual gradient inputs
KEY1, KEY2 = jax.random.split(jax.random.PRNGKey(0), num=2)
dummy_dy = jax.random.uniform(KEY1, rast_out_shp) 
dummy_ddb = jax.random.uniform(KEY2, rast_out_shp) 
if not JAX_RENDERER:    
    dummy_dy = torch.from_numpy(np.array(dummy_dy)).cuda()
    dummy_ddb = torch.from_numpy(np.array(dummy_ddb)).cuda()

# context variable for torch autograd testing with manual gradient input
TorchCtx = namedtuple('TorchCtx', ['saved_tensors', 'saved_grad_db'])

#----------------------
# TEST 1 : Rasterize
#----------------------
if JAX_RENDERER:
    print("\n\n---------------------TESTING JAX RASTERIZE---------------------\n\n")

    pos_clip_ja_jax = jnp.array(pos_clip_ja.cpu())
    pos_idx_jax = jnp.array(pos_idx.cpu())
    resolution = jnp.array(resolution)

    # jit with dummy input
    (rast_out, rast_out_db), rasterize_vjp = jax.vjp(jax_renderer.rasterize, 
                                                        jnp.zeros_like(pos_clip_ja_jax), 
                                                        jnp.ones_like(pos_idx_jax), 
                                                        resolution)

    # evaluate and time.
    start_time = time.time()
    (rast_out, rast_out_db), rasterize_vjp = jax.vjp(jax_renderer.rasterize, 
                                                        pos_clip_ja_jax, 
                                                        pos_idx_jax, 
                                                        resolution)
    pos_grads = rasterize_vjp((dummy_dy, 
                                dummy_ddb))[0]
    end_time = time.time()

    # print results
    print("JAX FWD (sum, min, max):", rast_out.sum().item(), rast_out.min().item(), rast_out.max().item())
    print("JAX BWD (sum, min, max):", pos_grads.sum().item(), pos_grads.min().item(), pos_grads.max().item())  # TODO this returns nans
    print(f"JAX rasterization (eval + grad): {(end_time - start_time)*1000} ms")

    # save viz
    b.viz.get_depth_image(rast_out[0][:,:,2]).save("img_jax.png")

else:
    print("\n\n---------------------TESTING TORCH RASTERIZE---------------------\n\n")

    rast_out_shp = (len(pos_clip_ja), resolution[0], resolution[1], 4)

    # evaluate and time.
    start_time = time.time()
    rast_out, rast_out_db  = dr.rasterize(torch_glctx, pos_clip_ja, pos_idx, resolution=resolution)
    ctx = TorchCtx(saved_tensors=(pos_clip_ja, pos_idx, rast_out), saved_grad_db=True)
    pos_grads = dr._rasterize_func.backward(ctx, dummy_dy, dummy_ddb)[1]   # 7 outputs; all are None except pos input
    end_time = time.time()

    # print results
    print("TORCH FWD (sum, min, max):", rast_out.sum().item(), rast_out.min().item(), rast_out.max().item())
    print("TORCH BWD (sum, min, max):", pos_grads.sum().item(), pos_grads.min().item(), pos_grads.max().item())
    print(f"Torch rasterization (eval + grad): {(end_time - start_time)*1000} ms")

    # save viz
    b.viz.get_depth_image(jnp.array(rast_out[0][:,:,2].cpu())).save("img_torch.png")

#----------------------
# TEST: Interpolate
#----------------------
if JAX_RENDERER:
    print("\n\n---------------------TESTING JAX INTERPOLATE---------------------\n\n")
    posw_jax = jnp.array(posw.cpu())

    # jit with dummy input
    jax_renderer.interpolate(jnp.zeros_like(posw_jax), jnp.zeros_like(rast_out), jnp.ones_like(pos_idx_jax))

    start_time = time.time()
    (gb_pos, dummy), interpolate_vjp = jax.vjp(jax_renderer.interpolate, 
                                                        posw_jax, 
                                                        rast_out, 
                                                        pos_idx_jax)
    g_attr, g_rast, _ = interpolate_vjp((dummy_dy, 
                        dummy))
    end_time = time.time()

    # print results
    print(f"JAX FWD (sum, min, max): {gb_pos.sum().item(), gb_pos.min().item(), gb_pos.max().item()}")
    print(f"JAX BWD (sum, min, max): g_attr={g_attr.sum().item(), g_attr.min().item(), g_attr.max().item()}\ng_rast={g_rast.sum().item(), g_rast.min().item(), g_rast.max().item()}")
    print(f"JAX interpolation: {(end_time - start_time)*1000} ms")

    # save viz
    b.viz.get_depth_image(gb_pos[0][:,:,2]).save("interpolate_jax.png")
    print("---------------------------------------------------------------\n\n")

else:
    print("\n\n---------------------TESTING TORCH INTERPOLATE---------------------\n\n")

    start_time = time.time()
    gb_pos, dummy = dr.interpolate(posw, rast_out, pos_idx)
    ctx = TorchCtx(saved_tensors=(posw, rast_out, pos_idx), saved_grad_db=None)
    grads = dr._interpolate_func.backward(ctx, dummy_dy, dummy)   # 6 outputs; all are None except pos input
    g_attr, g_rast = grads[0], grads[1]
    end_time = time.time()

    # print results
    print(f"TORCH FWD (sum, min, max): {gb_pos.sum().item(), gb_pos.min().item(), gb_pos.max().item()}")
    print(f"TORCH BWD (sum, min, max): g_attr={g_attr.sum().item(), g_attr.min().item(), g_attr.max().item()}\ng_rast={g_rast.sum().item(), g_rast.min().item(), g_rast.max().item()}")
    print(f"Torch interpolation: {(end_time - start_time)*1000} ms")

    # save viz
    b.viz.get_depth_image(jnp.array(gb_pos[0][:,:,2].cpu())).save("interpolate_torch.png")
    print("---------------------------------------------------------------\n\n")
