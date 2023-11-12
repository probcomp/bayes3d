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
        self.rasterize = jax.tree_util.Partial(self._rasterize, self)
    
    def interpolate(self, pos, tri, resolution):
        return _interpolate_fwd_custom_call(self, pos, tri, resolution)

    @functools.partial(jax.custom_vjp, nondiff_argnums=(0,3))
    def _rasterize(self_ctx, pos, tri, resolution):
        rast_out, rast_out_db = _rasterize_fwd_custom_call(self_ctx, pos, tri, resolution)
        return (rast_out, rast_out_db)

    def _rasterize_fwd(self_ctx, pos, tri, resolution):
        rast_out, rast_out_db = _rasterize_fwd_custom_call(self_ctx, pos, tri, resolution)
        saved_tensors = (pos, tri, rast_out, rast_out_db)

        return (rast_out, rast_out_db), saved_tensors

    def _rasterize_bwd(self_ctx, _resolution, saved_tensors, diffs):
        pos, tri, rast_out, rast_out_db = saved_tensors
        dy, ddb = diffs
        grads = _rasterize_bwd_custom_call(self_ctx, pos, tri, rast_out, dy, ddb)
        return grads[0], None

    _rasterize.defvjp(_rasterize_fwd, 
                    _rasterize_bwd)

# =========================
# Register custom call targets
# =========================

@functools.lru_cache
def _register_custom_calls():
    for _name, _value in dr._get_plugin(gl=True).registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="gpu")


# =========================
# Rasterize
# =========================
#### FORWARD ####
@functools.partial(jax.jit, static_argnums=(0,))
def _rasterize_fwd_custom_call(r: "Renderer", pos, tri, resolution):
    return _build_rasterize_fwd_primitive(r).bind(pos, tri, resolution)

@functools.lru_cache(maxsize=None)
def _build_rasterize_fwd_primitive(r: "Renderer"):
    _register_custom_calls()
    # For JIT compilation we need a function to evaluate the shape and dtype of the
    # outputs of our op for some given inputs

    def _rasterize_fwd_abstract(pos, tri, resolution):
        if (len(pos.shape) != 3):
            raise ValueError(f"Pass in a [num_images, num_vertices, 4] sized first input")
        num_images, num_vertices, _ = pos.shape
        num_triangles, _ = tri.shape

        dtype = dtypes.canonicalize_dtype(pos.dtype)

        if WITH_GRAD:
            return [ShapedArray((num_images, r.intrinsics.height, r.intrinsics.width, 4), dtype),
                    ShapedArray((num_images, r.intrinsics.height, r.intrinsics.width, 4), dtype)] 
        else:
            return [ShapedArray((num_images, r.intrinsics.height, r.intrinsics.width, 4), dtype)] 
        
    # Provide an MLIR "lowering" of the rasterize primitive.
    def _rasterize_fwd_lowering(ctx, pos, tri, resolution):
        """
        Single-object (one obj represented by tri) rasterization with 
        multiple poses (first dimension fo pos)
        dr.rasterize(glctx, pos, tri, resolution=resolution)
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
            raise NotImplementedError(f"Unsupported triangles dtype {triangles_aval.dtype}")

        num_images, num_vertices = poses_aval.shape[:2]
        num_triangles = triangles_aval.shape[0]
        out_shp_dtype = mlir.ir.RankedTensorType.get(
            [num_images, r.intrinsics.height, r.intrinsics.width, 4],
            mlir.dtype_to_ir_type(np_dtype))

        opaque = dr._get_plugin(gl=True).build_diff_rasterize_fwd_descriptor(r.renderer_env.cpp_wrapper,
                                                                    [num_images, num_vertices, num_triangles])

        op_name = "jax_rasterize_fwd_gl"

        if WITH_GRAD:
            return custom_call(
                op_name,
                # Output types
                result_types=[out_shp_dtype, out_shp_dtype],
                # The inputs:
                operands=[pos, tri, resolution],
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
                operands=[pos, tri, resolution],
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
    #     pass


    # *********************************************
    # *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
    # *********************************************
    # _render_prim = core.Primitive(f"render_multiple_{id(r)}")
    # _render_prim.multiple_results = True
    # _render_prim.def_impl(functools.partial(xla.apply_primitive, _render_prim))
    # _render_prim.def_abstract_eval(_render_abstract)
    _rasterize_prim = core.Primitive(f"rasterize_multiple_fwd_{id(r)}")
    _rasterize_prim.multiple_results = True
    _rasterize_prim.def_impl(functools.partial(xla.apply_primitive, _rasterize_prim))
    _rasterize_prim.def_abstract_eval(_rasterize_fwd_abstract)

    # # Connect the XLA translation rules for JIT compilation
    # mlir.register_lowering(_render_prim, _render_lowering, platform="gpu")
    mlir.register_lowering(_rasterize_prim, _rasterize_fwd_lowering, platform="gpu")


    # TODO add baching support
    # batching.primitive_batchers[_render_prim] = _render_batch

    return _rasterize_prim


#### BACKWARD ####
@functools.partial(jax.jit, static_argnums=(0,))
def _rasterize_bwd_custom_call(r: "Renderer", pos, tri, rast_out, dy, ddb):
    return _build_rasterize_bwd_primitive(r).bind(pos, tri, rast_out, dy, ddb)

@functools.lru_cache(maxsize=None)
def _build_rasterize_bwd_primitive(r: "Renderer"):
    _register_custom_calls()
    # For JIT compilation we need a function to evaluate the shape and dtype of the
    # outputs of our op for some given inputs

    def _rasterize_bwd_abstract(pos, tri, rast_out, dy, ddb):
        if (len(pos.shape) != 3):
            raise ValueError(f"Pass in a [num_images, num_vertices, 4] sized first input")
        out_shp = pos.shape
        dtype = dtypes.canonicalize_dtype(pos.dtype)

        print("abstract dtype=", dtype)

        return [ShapedArray(out_shp, dtype)]

    # Provide an MLIR "lowering" of the rasterize primitive.
    def _rasterize_bwd_lowering(ctx, pos, tri, rast_out, dy, ddb):
        # Extract the numpy type of the inputs
        pos_aval, tri_aval, rast_aval, dy_aval, ddb_aval = ctx.avals_in

        num_images, num_vertices = pos_aval.shape[:2]
        num_triangles = tri_aval.shape[0]
        depth, height, width = rast_aval.shape[:3]

        if rast_aval.ndim != 4:
            raise NotImplementedError(f"Rasterization output should be 4D: got {rast_aval.shape}")
        if dy_aval.ndim != 4 or ddb_aval.ndim != 4:
            raise NotImplementedError(f"Grad outputs from rasterize should be 4D: got dy={rast_out_aval.shape} and ddb={rast_out_aval.shape}")

        np_dtype = np.dtype(rast_aval.dtype)
        if np_dtype != np.float32:
            raise NotImplementedError(f"Unsupported dtype {np_dtype}")

        out_shp_dtype = mlir.ir.RankedTensorType.get(
            [num_images, num_vertices, 4],
            mlir.dtype_to_ir_type(np_dtype))  # gradients have same size as the positions

        opaque = dr._get_plugin(gl=True).build_diff_rasterize_bwd_descriptor([num_images, num_vertices], 
                                                                            [num_triangles], 
                                                                            [depth, height, width])

        op_name = "jax_rasterize_bwd"

        return custom_call(
            op_name,
            # Output types
            result_types=[out_shp_dtype],
            # The inputs:
            operands=[pos, tri, rast_out, dy, ddb],
            backend_config=opaque
        ).results



    # # ************************************
    # # *  SUPPORT FOR BATCHING WITH VMAP  *
    # # ************************************
    # def _render_batch(args, axes):
    #     pass

    # *********************************************
    # *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
    # *********************************************
    # _render_prim = core.Primitive(f"render_multiple_{id(r)}")
    # _render_prim.multiple_results = True
    # _render_prim.def_impl(functools.partial(xla.apply_primitive, _render_prim))
    # _render_prim.def_abstract_eval(_render_abstract)
    _rasterize_prim = core.Primitive(f"rasterize_multiple_bwd_{id(r)}")
    _rasterize_prim.multiple_results = True
    _rasterize_prim.def_impl(functools.partial(xla.apply_primitive, _rasterize_prim))
    _rasterize_prim.def_abstract_eval(_rasterize_bwd_abstract)

    # # Connect the XLA translation rules for JIT compilation
    # mlir.register_lowering(_render_prim, _render_lowering, platform="gpu")
    mlir.register_lowering(_rasterize_prim, _rasterize_bwd_lowering, platform="gpu")


    # TODO add baching support
    # batching.primitive_batchers[_render_prim] = _render_batch

    return _rasterize_prim

# =========================
# Interpolate
# =========================
@functools.partial(jax.jit, static_argnums=(0,))
def _interpolate_fwd_custom_call(r: "Renderer", attr, rast_out, tri):
    return _build_interpolate_fwd_primitive(r).bind(attr, rast_out, tri)

@functools.lru_cache(maxsize=None)
def _build_interpolate_fwd_primitive(r: "Renderer"):
    _register_custom_calls()
    # For JIT compilation we need a function to evaluate the shape and dtype of the
    # outputs of our op for some given inputs

    def _interpolate_fwd_abstract(attr, rast_out, tri):
        if (len(attr.shape) != 3):
            raise ValueError(f"Pass in a [num_images, num_vertices, num_attributes] sized first input")
        num_images, num_vertices, num_attributes = attr.shape
        _, height, width, _ = rast_out.shape
        num_tri, _ = tri.shape

        dtype = dtypes.canonicalize_dtype(attr.dtype)

        out_abstract = ShapedArray((num_images, height, width, num_attributes), dtype)
        out_db_abstract = ShapedArray((num_images, height, width, 0), dtype) # empty tensor
        return [out_abstract, out_db_abstract] 
        
    # Provide an MLIR "lowering" of the interpolate primitive.
    def _interpolate_fwd_lowering(ctx, attr, rast_out, tri):
        # Extract the numpy type of the inputs
        attr_aval, rast_out_aval, tri_aval = ctx.avals_in
        if attr_aval.ndim != 3:
            raise NotImplementedError(f"Only 3D attribute inputs supported: got {attr_aval.shape}")
        if rast_out_aval.ndim != 4:
            raise NotImplementedError(f"Only 4D rast inputs supported: got {rast_out_aval.shape}")
        if tri_aval.ndim != 2:
            raise NotImplementedError(f"Only 2D triangle tensors supported: got {tri_aval.shape}")

        np_dtype = np.dtype(attr_aval.dtype)
        if np_dtype != np.float32:
            raise NotImplementedError(f"Unsupported attributes dtype {np_dtype}")
        if np.dtype(tri_aval.dtype) != np.int32:
            raise NotImplementedError(f"Unsupported triangle dtype {tri_aval.dtype}")

        num_images, num_vertices, num_attributes = attr_aval.shape
        depth, height, width = rast_out_aval.shape[:3]
        num_triangles = tri_aval.shape[0]

        out_shp_dtype = mlir.ir.RankedTensorType.get(
            [num_images, height, width, num_attributes],
            mlir.dtype_to_ir_type(np_dtype))
        out_db_shp_dtype = mlir.ir.RankedTensorType.get(
            [num_images, height, width, 0],
            mlir.dtype_to_ir_type(np_dtype))

        opaque = dr._get_plugin(gl=True).build_diff_interpolate_descriptor(
                                                                        [num_images, num_vertices, num_attributes], 
                                                                        [depth, height, width], 
                                                                        [num_triangles]
                                                                        )

        op_name = "jax_interpolate_fwd"

        return custom_call(
            op_name,
            # Output types
            result_types=[out_shp_dtype, out_db_shp_dtype],
            # The inputs:
            operands=[attr, rast_out, tri],
            # # Layout specification:
            # operand_layouts=[(2, 1, 0,), (1, 0,), (0,)],
            # result_layouts=[(3, 2, 1, 0), (3, 2, 1, 0)],
            # GPU specific additional data
            backend_config=opaque
        ).results



    # # ************************************
    # # *  SUPPORT FOR BATCHING WITH VMAP  *
    # # ************************************
    # def _render_batch(args, axes):
    #     pass


    # *********************************************
    # *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
    # *********************************************
    # _render_prim = core.Primitive(f"render_multiple_{id(r)}")
    # _render_prim.multiple_results = True
    # _render_prim.def_impl(functools.partial(xla.apply_primitive, _render_prim))
    # _render_prim.def_abstract_eval(_render_abstract)
    _interpolate_prim = core.Primitive(f"interpolate_multiple_fwd_{id(r)}")
    _interpolate_prim.multiple_results = True
    _interpolate_prim.def_impl(functools.partial(xla.apply_primitive, _interpolate_prim))
    _interpolate_prim.def_abstract_eval(_interpolate_fwd_abstract)

    # # Connect the XLA translation rules for JIT compilation
    # mlir.register_lowering(_render_prim, _render_lowering, platform="gpu")
    mlir.register_lowering(_interpolate_prim, _interpolate_fwd_lowering, platform="gpu")


    # TODO add baching support
    # batching.primitive_batchers[_render_prim] = _render_batch

    return _interpolate_prim


