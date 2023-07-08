from typing import Tuple

import functools
import bayes3d.nvdiffrast.common as dr
import torch
from torch.utils import dlpack
import bayes3d.camera
import bayes3d as j
import bayes3d as b
import bayes3d.transforms_3d as t3d
import trimesh
import jax.numpy as jnp
import jax
from jax import core, dtypes
from jax.abstract_arrays import ShapedArray
from jax.interpreters import batching, mlir, xla
from jax.lib import xla_client
import numpy as np
import jax.dlpack
from jaxlib.hlo_helpers import custom_call
from tqdm import tqdm

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
transform_image_zeros_parallel = jax.vmap(transform_image_zeros, in_axes=(0,None))
transform_image_zeros_parallel_jit = jax.jit(transform_image_zeros_parallel)


# Useful reference for understanding the custom calls setup:
#   https://github.com/dfm/extending-jax

@functools.lru_cache
def _register_custom_calls():
    for _name, _value in dr._get_plugin(gl=True).registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="gpu")

@functools.partial(jax.jit, static_argnums=(0,))
def render_custom_call(r: "Renderer", poses, idx):
    return build_render_primitive(r).bind(poses, idx)


@functools.lru_cache(maxsize=None)
def build_render_primitive(r: "Renderer"):
    _register_custom_calls()
    # For JIT compilation we need a function to evaluate the shape and dtype of the
    # outputs of our op for some given inputs
    def _render_abstract(poses, indices):
        num_images = poses.shape[0]
        if poses.shape[1] != indices.shape[0]:
            raise ValueError(f"Mismatched #objects: {poses.shape}, {indices.shape}")
        dtype = dtypes.canonicalize_dtype(poses.dtype)
        return [ShapedArray((num_images, r.intrinsics.height, r.intrinsics.width, 4), dtype),
                ShapedArray((), dtype)]

    # Provide an MLIR "lowering" of the render primitive.
    def _render_lowering(ctx, poses, indices):

        # Extract the numpy type of the inputs
        poses_aval, indices_aval = ctx.avals_in
        if poses_aval.ndim != 4:
            raise NotImplementedError(f"Only 4D inputs supported: got {poses_aval.shape}")
        if indices_aval.ndim != 1:
            raise NotImplementedError(f"Only 1D inputs supported: got {indices_aval.shape}")

        np_dtype = np.dtype(poses_aval.dtype)
        if np_dtype != np.float32:
            raise NotImplementedError(f"Unsupported poses dtype {np_dtype}")
        if np.dtype(indices_aval.dtype) != np.int32:
            raise NotImplementedError(f"Unsupported indices dtype {indices_aval.dtype}")

        num_images, num_objects = poses_aval.shape[:2]
        out_shp_dtype = mlir.ir.RankedTensorType.get(
            [num_images, r.intrinsics.height, r.intrinsics.width, 4],
            mlir.dtype_to_ir_type(poses_aval.dtype))

        if num_objects != indices_aval.shape[0]:
            raise ValueError("Mismatched #objects in poses vs indices: "
                             f"{num_objects} vs {indices_aval.shape[0]}")
        opaque = dr._get_plugin(gl=True).build_rasterize_descriptor(r.renderer_env.cpp_wrapper,
                                                                    r.proj_list,
                                                                    [num_objects, num_images])

        scalar_dummy = mlir.ir.RankedTensorType.get([], mlir.dtype_to_ir_type(poses_aval.dtype))
        op_name = "jax_rasterize_fwd_gl"
        return custom_call(
            op_name,
            # Output types
            out_types=[out_shp_dtype, scalar_dummy],
            # The inputs:
            operands=[poses, indices],
            # Layout specification:
            operand_layouts=[(3, 2, 0, 1), (0,)],
            result_layouts=[(3, 2, 1, 0), ()],
            # GPU specific additional data
            backend_config=opaque
        )


    # ************************************
    # *  SUPPORT FOR BATCHING WITH VMAP  *
    # ************************************
    def _render_batch(args, axes):
        poses, indices = args 

        if poses.ndim != 5:
            raise NotImplementedError("Underlying primitive must operate on 4D poses.")  
     
        poses = jnp.moveaxis(poses, axes[0], 0)
        size_1 = poses.shape[0]
        size_2 = poses.shape[1]
        num_objects = poses.shape[2]
        poses = poses.reshape(size_1 * size_2, num_objects, 4, 4)

        if poses.shape[1] != indices.shape[0]:
            raise ValueError(f"Mismatched object counts: {poses.shape[0]} vs {indices.shape[0]}")
        if poses.shape[-2:] != (4, 4):
            raise ValueError(f"Unexpected poses shape: {poses.shape}")
        renders, dummy = render_custom_call(r, poses, indices)

        renders = renders.reshape(size_1, size_2, *renders.shape[1:])
        out_axes = 0, None
        return (renders, dummy), out_axes


    # *********************************************
    # *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
    # *********************************************
    _render_prim = core.Primitive(f"render_multiple_{id(r)}")
    _render_prim.multiple_results = True
    _render_prim.def_impl(functools.partial(xla.apply_primitive, _render_prim))
    _render_prim.def_abstract_eval(_render_abstract)

    # Connect the XLA translation rules for JIT compilation
    mlir.register_lowering(_render_prim, _render_lowering, platform="gpu")
    batching.primitive_batchers[_render_prim] = _render_batch
    
    return _render_prim

def setup_renderer(intrinsics, num_layers=1024):
    b.RENDERER = Renderer(intrinsics, num_layers=num_layers)
    

class Renderer(object):
    def __init__(self, intrinsics, num_layers=1024):
        self.intrinsics = intrinsics
        self.renderer_env = dr.RasterizeGLContext(intrinsics.height, intrinsics.width, output_db=False)
        self.proj_list = list(bayes3d.camera.open_gl_projection_matrix(
            intrinsics.height, intrinsics.width, 
            intrinsics.fx, intrinsics.fy, 
            intrinsics.cx, intrinsics.cy, 
            intrinsics.near, intrinsics.far
        ).reshape(-1))

        dr._get_plugin(gl=True).setup(
            self.renderer_env.cpp_wrapper,
            intrinsics.height, intrinsics.width, num_layers
        )

        self.meshes =[]
        self.mesh_names =[]
        self.model_box_dims = jnp.zeros((0,3))

    def add_mesh_from_file(self, mesh_filename, mesh_name=None, scaling_factor=1.0, force=None, center_mesh=True):
        mesh = trimesh.load(mesh_filename, force=force)
        self.add_mesh(mesh, mesh_name=mesh_name, scaling_factor=scaling_factor, center_mesh=center_mesh)

    def add_mesh(self, mesh, mesh_name=None, scaling_factor=1.0, center_mesh=True):
        
        if mesh_name is None:
            mesh_name = f"object_{len(self.meshes)}"
        
        mesh.vertices = mesh.vertices * scaling_factor
        
        bounding_box_dims, bounding_box_pose = bayes3d.utils.aabb(mesh.vertices)
        if center_mesh:
            mesh.vertices = mesh.vertices - bounding_box_pose[:3,3]

        self.meshes.append(mesh)
        self.mesh_names.append(mesh_name)

        self.model_box_dims = jnp.vstack(
            [
                self.model_box_dims,
                bounding_box_dims
            ]
        )

        vertices = np.array(mesh.vertices)
        vertices = np.concatenate([vertices, np.ones((*vertices.shape[:-1],1))],axis=-1)
        triangles = np.array(mesh.faces)
        dr._get_plugin(gl=True).load_vertices_fwd(
            self.renderer_env.cpp_wrapper, 
            torch.tensor(vertices.astype("f"), device='cuda'),
            torch.tensor(triangles.astype(np.int32), device='cuda'),
        )

    def render_many(self, poses, indices):
        images_jnp = render_custom_call(self, poses, indices)[0]
        return transform_image_zeros_parallel(images_jnp, self.intrinsics)

    def render(self, poses, indices):
        return self.render_many(poses[None,...], indices)[0]


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
    pixels = temp1 + jnp.array([intrinsics.cx, intrinsics.cy])
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
