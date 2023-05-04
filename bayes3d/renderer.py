import bayes3d.nvdiffrast.common as dr
import torch
import bayes3d.camera
import bayes3d as j
import bayes3d.transforms_3d as t3d
import trimesh
import jax.numpy as jnp
import jax
import numpy as np
import jax.dlpack

def transform_image_zeros(image_jnp, intrinsics):
    image_jnp_2 = jnp.concatenate(
        [
            j.t3d.unproject_depth(image_jnp[:,:,2], intrinsics),
            image_jnp[:,:,3:]
        ],
        axis=-1
    )
    return image_jnp_2
transform_image_zeros_jit = jax.jit(transform_image_zeros)
transform_image_zeros_parallel_jit = jax.jit(jax.vmap(transform_image_zeros, in_axes=(0,None)))


class Renderer(object):
    def __init__(self, intrinsics, num_layers=512):
        self.intrinsics = intrinsics
        self.renderer_env = dr.RasterizeGLContext(intrinsics.height, intrinsics.width, output_db=False)
        self.proj_list = list(bayes3d.camera.open_gl_projection_matrix(
            intrinsics.height, intrinsics.width, intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy, intrinsics.near, intrinsics.far
        ).reshape(-1))
        
        dr._get_plugin(gl=True).setup(
            self.renderer_env.cpp_wrapper,
            intrinsics.height, intrinsics.width, num_layers
        )

        self.meshes =[]
        self.mesh_names =[]
        self.model_box_dims = jnp.zeros((0,3))

    def add_mesh_from_file(self, mesh_filename, mesh_name=None, scaling_factor=1.0, force=None):
        mesh = trimesh.load(mesh_filename, force=force)
        self.add_mesh(mesh, mesh_name=mesh_name, scaling_factor=scaling_factor)

    def add_mesh(self, mesh, mesh_name=None, scaling_factor=1.0):
        
        if mesh_name is None:
            mesh_name = f"object_{len(self.meshes)}"
        
        mesh.vertices = mesh.vertices * scaling_factor
        self.meshes.append(mesh)
        self.mesh_names.append(mesh_name)

        self.model_box_dims = jnp.vstack(
            [
                self.model_box_dims,
                bayes3d.utils.aabb(mesh.vertices)[0]
            ]
        )

        vertices = np.array(mesh.vertices)
        vertices = np.concatenate([vertices, np.ones((*vertices.shape[:-1],1))],axis=-1)
        triangles = np.array(mesh.faces)
        dr._get_plugin(gl=True).load_vertices_fwd(
            self.renderer_env.cpp_wrapper, torch.tensor(vertices.astype("f"), device='cuda'),
            torch.tensor(triangles.astype(np.int32), device='cuda'),
        )

    def render_to_torch(self, poses, idx, on_object=0):
        poses_torch = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(poses))
        images_torch = dr._get_plugin(gl=True).rasterize_fwd_gl(self.renderer_env.cpp_wrapper, poses_torch, self.proj_list, idx, on_object)
        return images_torch

    def render_single_object(self, pose, idx):
        images_torch = self.render_to_torch(pose[None, None, :, :], [idx])
        images_jnp = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(images_torch[0]))
        return transform_image_zeros_jit(images_jnp, self.intrinsics)

    def render_parallel(self, poses, idx):
        images_torch = self.render_to_torch(poses[None, :, :, :], [idx])
        images_jnp = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(images_torch))
        return transform_image_zeros_parallel_jit(images_jnp, self.intrinsics)


    def render_multiobject(self, poses, indices):
        images_torch = self.render_to_torch(poses[:, None, :, :], indices)
        images_jnp = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(images_torch[0]))
        return transform_image_zeros_jit(images_jnp, self.intrinsics)


    def render_multiobject_parallel(self, poses, indices, on_object=0):
        images_torch = self.render_to_torch(poses, indices, on_object=on_object)
        images_jnp = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(images_torch))
        return transform_image_zeros_parallel_jit(images_jnp, self.intrinsics)

def render_point_cloud(point_cloud, intrinsics, pixel_smudge=0):
    transformed_cloud = point_cloud
    point_cloud = jnp.vstack([jnp.zeros((1, 3)), transformed_cloud])
    pixels = project_cloud_to_pixels(point_cloud, intrinsics)
    x, y = jnp.meshgrid(jnp.arange(intrinsics.width), jnp.arange(intrinsics.height))
    matches = (jnp.abs(x[:, :, None] - pixels[:, 0]) <= pixel_smudge) & (jnp.abs(y[:, :, None] - pixels[:, 1]) <= pixel_smudge)
    matches = matches * (intrinsics.far * 2.0 - point_cloud[:,-1][None, None, :])
    a = jnp.argmax(matches, axis=-1)    
    return point_cloud[a]

def render_point_cloud_batched(point_cloud, intrinsics, NUM_PER, pixel_smudge=0):
    all_images = []
    num_iters = jnp.ceil(point_cloud.shape[0] / NUM_PER).astype(jnp.int32)
    for i in tqdm(range(num_iters)):
        img = j.render_point_cloud(point_cloud[i*NUM_PER:i*NUM_PER+NUM_PER], intrinsics)
        img = img.at[img[:,:,2] < intrinsics.near].set(intrinsics.far)
        all_images.append(img)
    all_images_stack = jnp.stack(all_images,axis=-2)
    best = all_images_stack[:,:,:,2].argmin(-1)
    img = all_images_stack[
        np.arange(intrinsics.height)[:, None],
        np.arange(intrinsics.width)[None, :],
        best,
        :
    ]
    return img

def project_cloud_to_pixels(point_cloud, intrinsics):
    point_cloud_normalized = point_cloud / point_cloud[:, 2].reshape(-1, 1)
    temp1 = point_cloud_normalized[:, :2] * jnp.array([intrinsics.fx,intrinsics.fy])
    temp2 = temp1 + jnp.array([intrinsics.cx, intrinsics.cy])
    pixels = jnp.round(temp2) 
    return pixels

def get_masked_and_complement_image(depth_image, segmentation_image, segmentation_id, intrinsics):
    mask =  (segmentation_image == segmentation_id)
    masked_image = depth_image * mask + intrinsics.far * (1.0 - mask)
    complement_image = depth_image * (1.0 - mask) + intrinsics.far * mask
    return masked_image, complement_image


def splice_image_parallel(rendered_object_image, obs_image_complement):
    keep_masks = jnp.logical_or(
        (rendered_object_image[:,:,:,2] <= obs_image_complement[None, :,:, 2]) * 
        rendered_object_image[:,:,:,2] > 0.0
        ,
        (obs_image_complement[:,:,2] == 0)[None, ...]
    )[...,None]
    rendered_images = keep_masks * rendered_object_image + (1.0 - keep_masks) * obs_image_complement
    return rendered_images
