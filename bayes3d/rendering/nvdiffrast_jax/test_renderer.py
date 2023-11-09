import jax
import jax.numpy as jnp
import numpy as np
import os, sys
import os, sys
import torch
import bayes3d as b
from jax_renderer import Renderer as JaxRenderer
sys.path.append('/home/ubuntu/workspace/diff-dope')
import diffdope as dd

JAX_RENDERER = True
if JAX_RENDERER:
    r = JaxRenderer(intrinsics)
else:
    import bayes3d.rendering.nvdiffrast_full.nvdiffrast.torch as dr
    glctx = dr.RasterizeGLContext()

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

model_dir = os.path.join(b.utils.get_assets_dir(),"bop/ycbv/models")
idx = 14
mesh_path = os.path.join(model_dir,"obj_" + "{}".format(idx).rjust(6, '0') + ".ply")
m = b.utils.load_mesh(mesh_path)
m = b.utils.scale_mesh(m, 1.0/100.0)

#---------------------
# Load object
#---------------------

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
posw = torch.cat([pos, torch.ones([pos.shape[0], pos.shape[1], 1]).cuda()], axis=2)  # (xyz) -> (xyz1)
transform_mtx = torch.matmul(proj_cam, rot_mtx_44)  # transform = projection + pose rotation
pos_clip_ja = dd.xfm_points(pos, transform_mtx[None,...])  # transform points

#----------------------
# Rasterize
#----------------------
if JAX_RENDERER:
    pos_clip_ja_jax = jnp.array(pos_clip_ja.cpu())
    pos_idx_jax = jnp.array(pos_idx.cpu())
    resolution = jnp.array([200,200])
    ret = r.rasterize(pos_clip_ja_jax, pos_idx_jax, resolution=resolution)
    b.viz.get_depth_image(ret[0][0][:,:,2]).save("img_jax.png")
else:
    resolution = [200,200]
    ret = dr.rasterize(glctx, pos_clip_ja, pos_idx, resolution=resolution)
    b.viz.get_depth_image(jnp.array(ret[0][0][:,:,2].cpu())).save("img_torch.png")

from IPython import embed; embed()


