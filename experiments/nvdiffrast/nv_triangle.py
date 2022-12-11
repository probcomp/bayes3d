# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import imageio
import os
import numpy as np
import torch
import nvdiffrast.torch as dr
import sys
import trimesh
import jax3dp3.utils
import jax3dp3.viz
import jax3dp3.transforms_3d as t3d
import time
import os
from jax3dp3.viz import save_depth_image
import jax.numpy as jnp
import jax3dp3.camera
import jax3dp3.triangle_renderer
import jax3dp3.transforms_3d as t3d
import jax
import time
import trimesh

def tensor(*args, **kwargs):
    return torch.tensor(*args, device='cuda', **kwargs)

def zeros(size):
    return torch.zeros(size, device='cuda')


h, w = 200,200
cx = (h-1)/2
cy = (w-1)/2
fx = 200.0
fy = 200.0
near=0.05
far=10.0

# left,right, bottom, top, near,far = (-znear*cx / fx, znear*(w - cx) / fx,
#            -znear*cy / fy, znear*(h - cy) / fy, 
#             znear, zfar)


# proj = np.array([
#     [2*near/ (right-left), 0.0, (right + left)/(right - left), 0.0],
#     [0.0, 2*near / (top-bottom), (top+bottom)/(top-bottom), 0.0],
#     [0.0, 0.0, (far+near)/(far - near), 2*far*near/(far-near)],
#     [0.0, 0.0, -1.0, 0.0]
# ])

def orthographic_matrix(left, right, bottom, top, near, far):
    return np.array(
        (
            (2 / (right - left), 0, 0, -(right + left) / (right - left)),
            (0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)),
            (0, 0, -2 / (far - near), -(far + near) / (far - near)),
            (0, 0, 0, 1),
        )
    )

def projection_matrix(w, h, fx, fy, cx, cy, near, far):
    # transform from cv2 camera coordinates to opengl (flipping sign of y and z)
    view = np.eye(4)
    view[1:3] *= -1

    # see http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
    persp = np.zeros((4, 4))
    persp[0, 0] = fx
    persp[1, 1] = fy
    persp[0, 2] = cx
    persp[1, 2] = cy
    persp[2, 2:] = near + far, near * far
    persp[3, 2] = -1
    # transform the camera matrix from cv2 to opengl as well (flipping sign of y and z)
    persp[:2, 1:3] *= -1

    # The origin of the image is in the *center* of the top left pixel.
    # The orthographic matrix should map the whole image *area* into the opengl NDC, therefore the -.5 below:
    orth = orthographic_matrix(-0.5, w - 0.5, -0.5, h - 0.5, near, far)
    return orth @ persp @ view

proj = projection_matrix(w, h, fx, fy, cx, cy, near, far)

glctx = dr.RasterizeGLContext()

mesh = trimesh.load(os.path.join(jax3dp3.utils.get_assets_dir(),"bunny.obj"))
vertices_orig = np.array(mesh.vertices)
vertices = vertices_orig.copy()
# vertices[:,0] = vertices_orig[:,1]
# vertices[:,1] = vertices_orig[:,0]
pose = t3d.transform_from_pos(jnp.array([0.0, 0.0, 2.2]))
view_space_vertices = t3d.apply_transform(vertices, pose)
view_space_vertices_h = t3d.add_homogenous_ones(view_space_vertices)
clip_space_vertices = view_space_vertices_h.dot(proj.T)
clip_space_vertices = tensor([np.array(clip_space_vertices)])

tri = tensor(mesh.faces + 0 , dtype=torch.int32)



start = time.time()
rast, _ = dr.rasterize(glctx, clip_space_vertices, tri, resolution=[h,w], grad_db=False)
end = time.time()
print ("Time elapsed:", end - start)

jax3dp3.viz.save_depth_image(np.array(rast[0,:,:,3].cpu()), "tri.png",min=0.0, max=5000.0)


depth = rast[0,:,:,2] 
neg_mask = depth == 0
depth = 2 * near * far / (far + near - depth * (far - near))
depth[neg_mask] = 0

depth_img = depth
print('depth_img.max():');print(depth_img.max())
print('depth_img.min():');print(depth_img.min())
jax3dp3.viz.save_depth_image(np.array(depth_img.cpu()), "bunny.png",max=3.0)


import jax.numpy as jnp

rays = jax3dp3.camera.camera_rays_from_params(h, w, fx, fy, cx, cy)
mesh = trimesh.load(os.path.join(jax3dp3.utils.get_assets_dir(),"bunny.obj"))
trimesh_shape = mesh.vertices[mesh.faces]
img = jax3dp3.triangle_renderer.render_triangles(pose, trimesh_shape, rays)
save_depth_image(img[:,:,2], "triangle.png", max=3.0)

from IPython import embed; embed()
