import nvdiffrast.torch as dr
import trimesh
import numpy as np
import os
import jax3dp3.camera
import jax3dp3.utils
import torch
import jax3dp3.transforms_3d as t3d
import jax.numpy as jnp
import jax3dp3.viz
import time
def tensor(*args, **kwargs):
    return torch.tensor(*args, device='cuda', **kwargs)

def zeros(size):
    return torch.zeros(size, device='cuda')

h, w = 200,200
fx,fy = 200, 200
cx,cy = 100, 100
near=0.01
far=100.0


mesh = trimesh.load(os.path.join(jax3dp3.utils.get_assets_dir(),"bunny.obj"))
vertices_orig = np.array(mesh.vertices)
vertices = vertices_orig.copy()
pose = t3d.transform_from_pos(jnp.array([0.0, 0.0, 4.2]))
view_space_vertices = t3d.apply_transform(vertices, pose)
vertices = tensor(np.array([view_space_vertices]))
num_images = 1000
vertices = vertices.tile((num_images,1,1))
triangles = tensor(mesh.faces , dtype=torch.int32)

proj = tensor(jax3dp3.camera.open_gl_projection_matrix(h, w, fx, fy, cx, cy, near, far).astype("f"))
view_space_vertices_h = torch.concatenate([vertices, torch.ones((*vertices.shape[:-1],1) , device='cuda')],axis=-1)
clip_space_vertices = torch.einsum("ij,abj->abi", proj, view_space_vertices_h).contiguous()

glenv = dr.RasterizeGLContext()


start = time.time()
rast, _ = dr.rasterize(glenv, clip_space_vertices, triangles, resolution=[h,w], grad_db=False)
scores = rast[:,:,:,-1].sum((1,2))
best_idx = torch.argmax(scores)
print(clip_space_vertices[best_idx,0])
end = time.time()
print ("Time elapsed:", end - start)



jax3dp3.viz.save_depth_image(rast[0,:,:,2].cpu().numpy(), "bunny.png",max=20.0)