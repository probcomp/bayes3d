from typing import Tuple

import functools
import bayes3d.nvdiffrast.common as dr
import torch
from torch.utils import dlpack
import bayes3d.camera
import bayes3d as j
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
transform_image_zeros_parallel_jit = jax.jit(jax.vmap(transform_image_zeros, in_axes=(0,None)))


# Useful reference for understanding the custom calls setup:
#   https://github.com/dfm/extending-jax

@functools.lru_cache
def _register_custom_calls():
    for _name, _value in dr._get_plugin(gl=True).registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="gpu")


@functools.lru_cache(maxsize=None)
def build_setup_primitive(r: "Renderer", h, w, num_layers):
    _register_custom_calls()
    
    # For JIT compilation we need a function to evaluate the shape and dtype of the
    # outputs of our op for some given inputs
    def _setup_abstract():
        dtype = dtypes.canonicalize_dtype(np.float32)
        return [ShapedArray((), dtype), ShapedArray((), dtype)]

    # Provide an MLIR "lowering" of the load_vertices primitive.
    def _setup_lowering(ctx):

        opaque = dr._get_plugin(gl=True).build_setup_descriptor(
            r.renderer_env.cpp_wrapper, h, w, num_layers)

        scalar_dummy = mlir.ir.RankedTensorType.get([], mlir.dtype_to_ir_type(np.dtype(np.float32)))
        op_name = "jax_setup"
        return custom_call(
            op_name,
            # Output types
            out_types=[scalar_dummy, scalar_dummy],
            # The inputs:
            operands=[],
            # Layout specification:
            operand_layouts=[],
            result_layouts=[(), ()],
            # GPU specific additional data
            backend_config=opaque
        )

    # *********************************************
    # *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
    # *********************************************
    _prim = core.Primitive(f"setup__{id(r)}")
    _prim.multiple_results = True
    _prim.def_impl(functools.partial(xla.apply_primitive, _prim))
    _prim.def_abstract_eval(_setup_abstract)

    # Connect the XLA translation rules for JIT compilation
    mlir.register_lowering(_prim, _setup_lowering, platform="gpu")
    
    return _prim


@functools.lru_cache(maxsize=None)
def build_load_vertices_primitive(r: "Renderer"):
    _register_custom_calls()
    
    # For JIT compilation we need a function to evaluate the shape and dtype of the
    # outputs of our op for some given inputs
    def _load_vertices_abstract(vertices, triangles):
        dtype = dtypes.canonicalize_dtype(np.float32)
        return [ShapedArray((), dtype), ShapedArray((), dtype)]

    # Provide an MLIR "lowering" of the load_vertices primitive.
    def _load_vertices_lowering(ctx, vertices, triangles):
        # Extract the numpy type of the inputs
        vertices_aval, triangles_aval = ctx.avals_in

        if np.dtype(vertices_aval.dtype) != np.float32:
            raise NotImplementedError(f"Unsupported vertices dtype {np_dtype}")
        if np.dtype(triangles_aval.dtype) != np.int32:
            raise NotImplementedError(f"Unsupported triangles dtype {np_dtype}")

        opaque = dr._get_plugin(gl=True).build_load_vertices_descriptor(
            r.renderer_env.cpp_wrapper, vertices_aval.shape[0], triangles_aval.shape[0])

        scalar_dummy = mlir.ir.RankedTensorType.get([], mlir.dtype_to_ir_type(np.dtype(np.float32)))
        op_name = "jax_load_vertices"
        return custom_call(
            op_name,
            # Output types
            out_types=[scalar_dummy, scalar_dummy],
            # The inputs:
            operands=[vertices, triangles],
            # Layout specification:
            operand_layouts=[(1, 0), (1, 0)],
            result_layouts=[(), ()],
            # GPU specific additional data
            backend_config=opaque
        )

    # *********************************************
    # *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
    # *********************************************
    _prim = core.Primitive(f"load_vertices__{id(r)}")
    _prim.multiple_results = True
    _prim.def_impl(functools.partial(xla.apply_primitive, _prim))
    _prim.def_abstract_eval(_load_vertices_abstract)

    # Connect the XLA translation rules for JIT compilation
    mlir.register_lowering(_prim, _load_vertices_lowering, platform="gpu")
    
    return _prim


@functools.partial(jax.jit, static_argnums=(0, 3))
def render_custom_call(r: "Renderer", poses, idx, on_object=0):
    return build_render_primitive(r, int(on_object)).bind(poses, idx)


@functools.lru_cache(maxsize=None)
def build_render_primitive(r: "Renderer", on_object: int = 0):
    _register_custom_calls()
    
    # For JIT compilation we need a function to evaluate the shape and dtype of the
    # outputs of our op for some given inputs
    def _render_abstract(poses, indices):
        num_images = poses.shape[1]
        if poses.shape[0] != indices.shape[0]:
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

        num_objects, num_images = poses_aval.shape[:2]
        out_shp_dtype = mlir.ir.RankedTensorType.get(
            [num_images, r.intrinsics.height, r.intrinsics.width, 4],
            mlir.dtype_to_ir_type(poses_aval.dtype))

        if num_objects != indices_aval.shape[0]:
            raise ValueError("Mismatched #objects in poses vs indices: "
                             f"{num_objects} vs {indices_aval.shape[0]}")
        opaque = dr._get_plugin(gl=True).build_rasterize_descriptor(r.renderer_env.cpp_wrapper,
                                                                    r.proj_list,
                                                                    [num_objects, num_images],
                                                                    on_object)

        scalar_dummy = mlir.ir.RankedTensorType.get([], mlir.dtype_to_ir_type(poses_aval.dtype))
        op_name = "jax_rasterize_fwd_gl"
        return custom_call(
            op_name,
            # Output types
            out_types=[out_shp_dtype, scalar_dummy],
            # The inputs:
            operands=[poses, indices],
            # Layout specification:
            operand_layouts=[(3, 2, 1, 0), (0,)],
            result_layouts=[(3, 2, 1, 0), ()],
            # GPU specific additional data
            backend_config=opaque
        )
    
    # ************************************
    # *  SUPPORT FOR BATCHING WITH VMAP  *
    # ************************************

    # Our op already supports arbitrary dimensions so the batching rule is quite
    # simple. The jax.lax.linalg module includes some example of more complicated
    # batching rules if you need such a thing.
    def _render_batch(args, axes):
        poses, indices = args
        if axes[1] is not None:
            raise NotImplementedError("Batching on object indices is not yet supported.")
        if axes[0] is None:
            raise NotImplementedError("Must batch a dimension of poses.")
        if poses.ndim != 5:
            raise NotImplementedError("Underlying primitive must operate on 4D poses.")
        # 4D pose shape is [num_objects, num_images, 4, 4].
        orig_num_images = poses.shape[1]
        num_batched = poses.shape[axes[0]]
        new_num_images = num_batched * orig_num_images
        # First, we will move the batched axis to be adjacent to num_images (axis 1).
        poses = jnp.moveaxis(poses, axes[0], 1)
        poses = poses.reshape(poses.shape[0], new_num_images, 4, 4)
        if poses.shape[0] != indices.shape[0]:
            raise ValueError(f"Mismatched object counts: {poses.shape[0]} vs {indices.shape[0]}")
        if poses.shape[-2:] != (4, 4):
            raise ValueError(f"Unexpected poses shape: {poses.shape}")
        renders, dummy = render_custom_call(r, poses, indices, on_object=on_object)
        renders = renders.reshape(num_batched, orig_num_images, *renders.shape[1:])
        out_axes = 0, None
        return (renders, dummy), out_axes

    # *********************************************
    # *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
    # *********************************************
    _render_prim = core.Primitive(f"render__{id(r)}__{on_object}")
    _render_prim.multiple_results = True
    _render_prim.def_impl(functools.partial(xla.apply_primitive, _render_prim))
    _render_prim.def_abstract_eval(_render_abstract)

    # Connect the XLA translation rules for JIT compilation
    mlir.register_lowering(_render_prim, _render_lowering, platform="gpu")
    batching.primitive_batchers[_render_prim] = _render_batch
    
    return _render_prim


CUSTOM_CALLS = True


class Renderer(object):
    def __init__(self, intrinsics, num_layers=512):
        self.intrinsics = intrinsics
        self.renderer_env = dr.RasterizeGLContext(intrinsics.height, intrinsics.width, output_db=False)
        self.proj_list = list(bayes3d.camera.open_gl_projection_matrix(
            intrinsics.height, intrinsics.width, 
            intrinsics.fx, intrinsics.fy, 
            intrinsics.cx, intrinsics.cy, 
            intrinsics.near, intrinsics.far
        ).reshape(-1))

        if not CUSTOM_CALLS:
            dr._get_plugin(gl=True).setup(
                self.renderer_env.cpp_wrapper,
                intrinsics.height, intrinsics.width, num_layers
            )
        else:
            build_setup_primitive(self,intrinsics.height, intrinsics.width, num_layers).bind()

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
        if CUSTOM_CALLS:
            prim = build_load_vertices_primitive(self)
            prim.bind(jnp.float32(vertices), jnp.int32(triangles))
        else:
            dr._get_plugin(gl=True).load_vertices_fwd(
                self.renderer_env.cpp_wrapper, 
                torch.tensor(vertices.astype("f"), device='cuda'),
                torch.tensor(triangles.astype(np.int32), device='cuda'),
            )

    def render_to_torch(self, poses, idx, on_object=0):
        poses_torch = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(poses))
        images_torch = dr._get_plugin(gl=True).rasterize_fwd_gl(
            self.renderer_env.cpp_wrapper, poses_torch, self.proj_list, idx, on_object)
        return images_torch
    
    def render_jax(self, poses, idx, on_object=0):
        poses = jnp.float32(poses)
        idx = jnp.int32(idx)
        return render_custom_call(self, poses, idx, on_object)[0]
        
    def render_single_object(self, pose, idx):
        if not CUSTOM_CALLS:
            images_torch = self.render_to_torch(pose[None, None, :, :], [idx])
            images_jnp = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(images_torch[0]))
        else:
            images_jnp = self.render_jax(pose[None, None, :, :], [idx])[0]
        return transform_image_zeros_jit(images_jnp, self.intrinsics)

    def render_parallel(self, poses, idx):
        if not CUSTOM_CALLS:
            images_torch = self.render_to_torch(poses[None, :, :, :], [idx])
            images_jnp = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(images_torch))
        else:
            images_jnp = self.render_jax(poses[None, :, :, :], [idx])
        return transform_image_zeros_parallel_jit(images_jnp, self.intrinsics)


    def render_multiobject(self, poses, indices):
        if not CUSTOM_CALLS:
            images_torch = self.render_to_torch(poses[:, None, :, :], indices)
            images_jnp = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(images_torch[0]))
        else:
            images_jnp = self.render_jax(poses[:, None, :, :], indices)[0]
        return transform_image_zeros_jit(images_jnp, self.intrinsics)


    def render_multiobject_parallel(self, poses, indices, on_object=0):
        if not CUSTOM_CALLS:
            images_torch = self.render_to_torch(poses, indices, on_object=on_object)
            images_jnp = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(images_torch))
        else:
            images_jnp = self.render_jax(poses, indices, on_object=on_object)
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
