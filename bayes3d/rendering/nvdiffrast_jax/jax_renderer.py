from typing import Tuple

import functools
# import bayes3d._rendering.nvdiffrast.common as dr
import bayes3d.rendering.nvdiffrast_jax.nvdiffrast.jax as dr
import bayes3d.camera
import bayes3d as j
import bayes3d as b
import bayes3d.transforms_3d as t3d
import trimesh
import jax.numpy as jnp
import jax
from jax import core, dtypes
from jax.core import ShapedArray
from jax.interpreters import batching, mlir, xla
from jax.lib import xla_client
import numpy as np
from jaxlib.hlo_helpers import custom_call
from tqdm import tqdm

WITH_GRAD = True # todo move to renderer constructor

class Renderer(object):
    def __init__(self, intrinsics, num_layers=1024):
        """A renderer for rendering meshes.
        
        Args:
            intrinsics (bayes3d.camera.Intrinsics): The camera intrinsics.
            num_layers (int, optional): The number of scenes to render in parallel. Defaults to 1024.
        """
        self.intrinsics = intrinsics
        self.renderer_env = dr.RasterizeGLContext(output_db=WITH_GRAD)
        self.proj_list = list(bayes3d.camera._open_gl_projection_matrix(
            intrinsics.height, intrinsics.width, 
            intrinsics.fx, intrinsics.fy, 
            intrinsics.cx, intrinsics.cy, 
            intrinsics.near, intrinsics.far
        ).reshape(-1))

    def rasterize(self, pos_clip_ja, pos_idx, resolution):
        return _render_custom_call(self, pos_clip_ja, pos_idx, resolution)

# Useful reference for understanding the custom calls setup:
#   https://github.com/dfm/extending-jax

@functools.lru_cache
def _register_custom_calls():
    for _name, _value in dr._get_plugin(gl=True).registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="gpu")

@functools.partial(jax.jit, static_argnums=(0,))
def _render_custom_call(r: "Renderer", pos_clip_ja, pos_idx, resolution):
    return _build_render_primitive(r).bind(pos_clip_ja, pos_idx, resolution)

@functools.lru_cache(maxsize=None)
def _build_render_primitive(r: "Renderer"):
    _register_custom_calls()
    # For JIT compilation we need a function to evaluate the shape and dtype of the
    # outputs of our op for some given inputs

    def _rasterize_abstract(pos_clip_ja, pos_idx, resolution):
        if (len(pos_clip_ja.shape) != 3):
            raise ValueError(f"Pass in a [num_images, num_vertices, 4] sized first input")
        num_images, num_vertices, _ = pos_clip_ja.shape
        num_triangles, _ = pos_idx.shape

        dtype = dtypes.canonicalize_dtype(pos_clip_ja.dtype)

        if WITH_GRAD:
            return [ShapedArray((num_images, r.intrinsics.height, r.intrinsics.width, 4), dtype),
                    ShapedArray((num_images, r.intrinsics.height, r.intrinsics.width, 4), dtype)] 
        else:
            return [ShapedArray((num_images, r.intrinsics.height, r.intrinsics.width, 4), dtype)] 
        
    # Provide an MLIR "lowering" of the rasterize primitive.
    def _rasterize_lowering(ctx, pos_clip_ja, pos_idx, resolution):
        """
        Single-object (one obj represented by pos_idx) rasterization with 
        multiple poses (first dimension fo pos_clip_ja)
        dr.rasterize(glctx, pos_clip_ja, pos_idx, resolution=resolution)
        """
        # Extract the numpy type of the inputs
        poses_aval, triangles_aval, resolution_aval = ctx.avals_in
        if poses_aval.ndim != 3:
            raise NotImplementedError(f"Only 3D vtx position inputs supported: got {poses_aval.shape}")
        if triangles_aval.ndim != 2:
            raise NotImplementedError(f"Only 2D triangle inputs supported: got {triangles_aval.shape}")
        if resolution_aval.shape[0] != 2:
            raise NotImplementedError(f"Only 2D resolutions supported: got {resolution_aval.shape}")

        np_dtype = np.dtype(poses_aval.dtype)
        if np_dtype != np.float32:
            raise NotImplementedError(f"Unsupported vtx positions dtype {np_dtype}")
        if np.dtype(triangles_aval.dtype) != np.int32:
            raise NotImplementedError(f"Unsupported triangle triangles dtype {triangles_aval.dtype}")

        num_images, num_vertices = poses_aval.shape[:2]
        num_triangles = triangles_aval.shape[0]
        out_shp_dtype = mlir.ir.RankedTensorType.get(
            [num_images, r.intrinsics.height, r.intrinsics.width, 4],
            mlir.dtype_to_ir_type(np_dtype))

        opaque = dr._get_plugin(gl=True).build_diff_rasterize_descriptor(r.renderer_env.cpp_wrapper,
                                                                    [num_images, num_vertices, num_triangles])

        op_name = "jax_rasterize_fwd_gl"

        if WITH_GRAD:
            return custom_call(
                op_name,
                # Output types
                result_types=[out_shp_dtype, out_shp_dtype],
                # The inputs:
                operands=[pos_clip_ja, pos_idx, resolution],
                # # Layout specification:
                # operand_layouts=[(2, 1, 0,), (1, 0,), (0,)],
                # result_layouts=[(3, 2, 1, 0), (3, 2, 1, 0)],
                # GPU specific additional data
                backend_config=opaque
            ).results
        else:
            return custom_call(
                op_name,
                # Output types
                result_types=[out_shp_dtype],
                # The inputs:
                operands=[pos_clip_ja, pos_idx, resolution],
                # # Layout specification:
                operand_layouts=[(2, 1, 0,), (1, 0,), (0,)],
                result_layouts=[(3, 2, 1, 0), ],
                # GPU specific additional data
                backend_config=opaque
            ).results


    # # ************************************
    # # *  SUPPORT FOR BATCHING WITH VMAP  *
    # # ************************************
    # def _render_batch(args, axes):
    #     poses, indices = args 

    #     if poses.ndim != 5:
    #         raise NotImplementedError("Underlying primitive must operate on 4D poses.")  
     
    #     poses = jnp.moveaxis(poses, axes[0], 0)
    #     size_1 = poses.shape[0]
    #     size_2 = poses.shape[1]
    #     num_objects = poses.shape[2]
    #     poses = poses.reshape(size_1 * size_2, num_objects, 4, 4)

    #     if poses.shape[1] != indices.shape[0]:
    #         raise ValueError(f"Mismatched object counts: {poses.shape[0]} vs {indices.shape[0]}")
    #     if poses.shape[-2:] != (4, 4):
    #         raise ValueError(f"Unexpected poses shape: {poses.shape}")
    #     renders, dummy = _render_custom_call(r, poses, indices)

    #     renders = renders.reshape(size_1, size_2, *renders.shape[1:])
    #     out_axes = 0, None
    #     return (renders, dummy), out_axes


    # *********************************************
    # *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
    # *********************************************
    # _render_prim = core.Primitive(f"render_multiple_{id(r)}")
    # _render_prim.multiple_results = True
    # _render_prim.def_impl(functools.partial(xla.apply_primitive, _render_prim))
    # _render_prim.def_abstract_eval(_render_abstract)
    _rasterize_prim = core.Primitive(f"rasterize_multiple_{id(r)}")
    _rasterize_prim.multiple_results = True
    _rasterize_prim.def_impl(functools.partial(xla.apply_primitive, _rasterize_prim))
    _rasterize_prim.def_abstract_eval(_rasterize_abstract)

    # # Connect the XLA translation rules for JIT compilation
    # mlir.register_lowering(_render_prim, _render_lowering, platform="gpu")
    mlir.register_lowering(_rasterize_prim, _rasterize_lowering, platform="gpu")


    # TODO add baching support
    # batching.primitive_batchers[_render_prim] = _render_batch

    return _rasterize_prim




