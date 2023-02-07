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

def setup_renderer(h, w, fx, fy, cx, cy, near, far, num_layers=2048):
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

def render_to_torch(poses, idx, on_object=0):
    poses_torch = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(poses))
    images_torch = dr._get_plugin(gl=True).rasterize_fwd_gl(RENDERER_ENV.cpp_wrapper, poses_torch, PROJ_LIST, idx, on_object)
    return images_torch

def render_single_object(pose, idx):
    images_torch = render_to_torch(pose[None, None, :, :], [idx])
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(images_torch[0]))

def render_parallel(poses, idx):
    images_torch = render_to_torch(poses[None, :, :, :], [idx])
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(images_torch))

def render_multiobject(poses, indices):
    images_torch = render_to_torch(poses[:, None, :, :], indices)
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(images_torch[0]))

def render_multiobject_parallel(poses, indices, on_object=0):
    images_torch = render_to_torch(poses, indices, on_object=on_object)
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(images_torch))

def render_point_cloud(point_cloud, h, w, fx,fy,cx,cy, near, far, pixel_smudge):
    transformed_cloud = point_cloud
    point_cloud = jnp.vstack([jnp.zeros((1, 3)), transformed_cloud])
    pixels = project_cloud_to_pixels(point_cloud, fx,fy,cx,cy)
    x, y = jnp.meshgrid(jnp.arange(w), jnp.arange(h))
    matches = (jnp.abs(x[:, :, None] - pixels[:, 0]) <= pixel_smudge) & (jnp.abs(y[:, :, None] - pixels[:, 1]) <= pixel_smudge)
    matches = matches * (far - point_cloud[:,-1][None, None, :])
    a = jnp.argmax(matches, axis=-1)    
    return point_cloud[a]
    
def project_cloud_to_pixels(point_cloud, fx,fy,cx,cy):
    point_cloud_normalized = point_cloud / point_cloud[:, 2].reshape(-1, 1)
    temp1 = point_cloud_normalized[:, :2] * jnp.array([fx,fy])
    temp2 = temp1 + jnp.array([cx, cy])
    pixels = jnp.round(temp2) 
    return pixels

def get_image_masked(point_cloud_image, segmentation_image, segmentation_id):
    mask =  (segmentation_image == segmentation_id)[:,:,None]
    image_masked = point_cloud_image * mask
    return image_masked

def get_image_masked_and_complement(point_cloud_image, segmentation_image, segmentation_id, far):
    mask =  (segmentation_image == segmentation_id)[:,:,None]
    image_masked = point_cloud_image * mask
    image_masked_complement = point_cloud_image * (1.0 - mask) + mask * far
    return image_masked, image_masked_complement

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


def splice_in_object_parallel(rendered_object_image, obs_image_complement):
    keep_masks = jnp.logical_or(
        (rendered_object_image[:,:,:,2] <= obs_image_complement[None, :,:, 2]) * 
        rendered_object_image[:,:,:,2] > 0.0
        ,
        (obs_image_complement[:,:,2] == 0)[None, ...]
    )[...,None]
    rendered_images = keep_masks * rendered_object_image + (1.0 - keep_masks) * obs_image_complement
    return rendered_images