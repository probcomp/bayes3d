import jax3dp3.nvdiffrast.common as dr
import torch
import jax3dp3.camera
import trimesh
import jax
import numpy as np
import jax.dlpack

RENDERER_ENV = None
PROJ_LIST = None

def setup_renderer(h, w, fx, fy, cx, cy, near, far, num_layers=1024):
    global RENDERER_ENV
    global PROJ_LIST
    RENDERER_ENV = dr.RasterizeGLContext(h, w, output_db=False)
    PROJ_LIST = list(jax3dp3.camera.open_gl_projection_matrix(h, w, fx, fy, cx, cy, near, far).reshape(-1))
    dr._get_plugin(gl=True).setup(
        RENDERER_ENV.cpp_wrapper,
        h,w, num_layers
    )

def load_model(mesh):
    vertices = np.array(mesh.vertices)
    vertices = np.concatenate([vertices, np.ones((*vertices.shape[:-1],1))],axis=-1)
    triangles = np.array(mesh.faces)
    dr._get_plugin(gl=True).load_vertices_fwd(
        RENDERER_ENV.cpp_wrapper, torch.tensor(vertices.astype("f"), device='cuda'),
        torch.tensor(triangles.astype(np.int32), device='cuda'),
    )

def render_to_torch(poses, idx):
    poses_torch = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(poses))
    images_torch = dr._get_plugin(gl=True).rasterize_fwd_gl(RENDERER_ENV.cpp_wrapper, poses_torch, PROJ_LIST, idx)
    return images_torch

def render_single_object(pose, idx):
    images_torch = render_to_torch(pose[None, None, :, :], [idx])
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(images_torch[0]))

def render_parallel(poses, idx):
    images_torch = render_to_torch(poses[:, None, :, :], [idx])
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(images_torch))

def render_multiobject(poses, indices):
    images_torch = render_to_torch(poses[None, :, :, :], indices)
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(images_torch[0]))

def render_multiobject_parallel(poses, indices):
    images_torch = render_to_torch(poses, indices)
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(images_torch))
