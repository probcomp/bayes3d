import jax3dp3.nvdiffrast.common as dr
import torch
import jax3dp3.camera
import trimesh
import jax.numpy as jnp
import jax
import numpy as np
import jax.dlpack
import cv2

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



# Complement rendering function
def combine_rendered_with_groud_truth(rendered_image, gt_point_cloud_image):
    keep_gt = jnp.logical_or(
        rendered_image[:,:,2] == 0.0,
        (
            (gt_point_cloud_image[:,:,2] != 0.0) *
            (rendered_image[:,:,2] >= gt_point_cloud_image[:,:,2])
        )
    )[...,None]

    images_apply_occlusions = (
        rendered_image[:,:,:3] * (1- keep_gt) + 
        gt_point_cloud_image * keep_gt
    )
    return images_apply_occlusions

def get_complement_masked_images(images_unmasked, gt_img_complement):
    blocked = images_unmasked[:,:,:,2] >= gt_img_complement[None,:,:,2] 
    nonzero = gt_img_complement[None, :, :, 2] != 0

    images = images_unmasked * (1-(blocked * nonzero))[:,:,:, None]  # rendered model images
    return images

def get_complement_masked_image(image_unmasked, gt_img_complement):
    blocked = image_unmasked[:,:,2] >= gt_img_complement[:,:,2] 
    nonzero = gt_img_complement[:, :, 2] != 0

    image = image_unmasked * (1-(blocked * nonzero))[:,:,None] # rendered model image
    return image
