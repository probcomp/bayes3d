import functools

import jax
import jax.numpy as jnp
import numpy as np
from jax import core, dtypes
from jax.core import ShapedArray
from jax.interpreters import mlir, xla
from jax.lib import xla_client
from jaxlib.hlo_helpers import custom_call

import bayes3d as b
import bayes3d.camera

# import bayes3d._rendering.nvdiffrast.common as dr
import bayes3d.rendering.nvdiffrast_jax.nvdiffrast.jax as dr


class Renderer(object):
    def __init__(self, intrinsics, num_layers=1024):
        """A renderer for rendering meshes.

        Args:
            intrinsics (bayes3d.camera.Intrinsics): The camera intrinsics.
            num_layers (int, optional): The number of scenes to render in parallel. Defaults to 1024.
        """
        self.intrinsics = intrinsics
        self.renderer_env = dr.RasterizeGLContext(output_db=True)
        self.rasterize = jax.tree_util.Partial(self._rasterize, self)
        self.interpolate = jax.tree_util.Partial(self._interpolate, self)

    # ------------------
    # Rasterization
    # ------------------

    @functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
    def _rasterize(self, pose, pos, tri, ranges, projMatrix, resolution):
        return _rasterize_fwd_custom_call(self, pose, pos, tri, ranges, projMatrix, resolution)

    def _rasterize_fwd(self, pose, pos, tri, ranges, projMatrix, resolution):
        rast_out, rast_out_db = _rasterize_fwd_custom_call(self, pose, pos, tri, ranges, projMatrix, resolution)
        saved_tensors = (pose, pos, tri, rast_out)
        return (rast_out, rast_out_db), saved_tensors

    def _rasterize_bwd(self, saved_tensors, diffs):
        pose, pos, tri, rast_out = saved_tensors
        dy, ddb = diffs

        grads = _rasterize_bwd_custom_call(self, pos, tri, rast_out, dy, ddb)
        return grads[0], None, None

    _rasterize.defvjp(_rasterize_fwd, _rasterize_bwd)

    # ------------------
    # Interpolation
    # ------------------

    @functools.partial(jax.custom_vjp, nondiff_argnums=(0,))
    def _interpolate(self, attr, rast, tri, rast_db, diff_attrs):
        num_total_attrs = attr.shape[-1]
        diff_attrs_all = jax.lax.cond(
            diff_attrs.shape[0] == num_total_attrs, lambda: True, lambda: False
        )
        return _interpolate_fwd_custom_call(
            self, attr, rast, tri, rast_db, diff_attrs_all, diff_attrs
        )

    def _interpolate_fwd(self, attr, rast, tri, rast_db, diff_attrs):
        num_total_attrs = attr.shape[-1]
        diff_attrs_all = jax.lax.cond(
            diff_attrs.shape[0] == num_total_attrs, lambda: True, lambda: False
        )
        out, out_da = _interpolate_fwd_custom_call(
            self, attr, rast, tri, rast_db, diff_attrs_all, diff_attrs
        )
        saved_tensors = (attr, rast, tri, rast_db, diff_attrs_all, diff_attrs)

        return (out, out_da), saved_tensors

    def _interpolate_bwd(self, saved_tensors, diffs):
        attr, rast, tri, rast_db, diff_attrs_all, diff_attrs_list = saved_tensors
        dy, dda = diffs
        g_attr, g_rast, g_rast_db = _interpolate_bwd_custom_call(
            self, attr, rast, tri, dy, rast_db, dda, diff_attrs_all, diff_attrs_list
        )
        return g_attr, g_rast, None, g_rast_db, None

    _interpolate.defvjp(_interpolate_fwd, _interpolate_bwd)

    # def render_many(self, vertices, faces, poses, intrinsics):
    #     jax_renderer = self
    #     projection_matrix = b.camera._open_gl_projection_matrix(
    #         intrinsics.height,
    #         intrinsics.width,
    #         intrinsics.fx,
    #         intrinsics.fy,
    #         intrinsics.cx,
    #         intrinsics.cy,
    #         intrinsics.near,
    #         intrinsics.far,
    #     )
    #     composed_projection = projection_matrix @ poses
    #     vertices_homogenous = jnp.concatenate(
    #         [vertices, jnp.ones((*vertices.shape[:-1], 1))], axis=-1
    #     )
    #     clip_spaces_projected_vertices = jnp.einsum(
    #         "nij,mj->nmi", composed_projection, vertices_homogenous
    #     )
    #     rast_out, rast_out_db = jax_renderer.rasterize(
    #         clip_spaces_projected_vertices,
    #         faces,
    #         jnp.array([intrinsics.height, intrinsics.width]),
    #     )
    #     interpolated_collided_vertices_clip, _ = jax_renderer.interpolate(
    #         jnp.tile(vertices_homogenous[None, ...], (poses.shape[0], 1, 1)),
    #         rast_out,
    #         faces,
    #         rast_out_db,
    #         jnp.array([0, 1, 2, 3]),
    #     )
    #     interpolated_collided_vertices = jnp.einsum(
    #         "a...ij,a...j->a...i", poses, interpolated_collided_vertices_clip
    #     )
    #     mask = rast_out[..., -1] > 0
    #     depth = interpolated_collided_vertices[..., 2] * mask
    #     return depth

    # def render(self, vertices, faces, object_pose, intrinsics):
    #     jax_renderer = self
    #     projection_matrix = b.camera._open_gl_projection_matrix(
    #         intrinsics.height,
    #         intrinsics.width,
    #         intrinsics.fx,
    #         intrinsics.fy,
    #         intrinsics.cx,
    #         intrinsics.cy,
    #         intrinsics.near,
    #         intrinsics.far,
    #     )
    #     final_mtx_proj = projection_matrix @ object_pose
    #     posw = jnp.concatenate([vertices, jnp.ones((*vertices.shape[:-1], 1))], axis=-1)
    #     pos_clip_ja = xfm_points(vertices, final_mtx_proj)
    #     rast_out, rast_out_db = jax_renderer.rasterize(
    #         pos_clip_ja[None, ...],
    #         faces,
    #         jnp.array([intrinsics.height, intrinsics.width]),
    #     )
    #     gb_pos, _ = jax_renderer.interpolate(
    #         posw[None, ...], rast_out, faces, rast_out_db, jnp.array([0, 1, 2, 3])
    #     )
    #     mask = rast_out[..., -1] > 0
    #     shape_keep = gb_pos.shape
    #     gb_pos = gb_pos.reshape(shape_keep[0], -1, shape_keep[-1])
    #     gb_pos = gb_pos[..., :3]
    #     depth = xfm_points(gb_pos, object_pose)
    #     depth = depth.reshape(shape_keep)[..., 2] * -1
    #     return -(depth * mask), mask


# ================================================================================================
# Register custom call targets helpers
# ================================================================================================
def xfm_points(points, matrix):
    points2 = jnp.concatenate([points, jnp.ones((*points.shape[:-1], 1))], axis=-1)
    return jnp.matmul(points2, matrix.T)


# XLA array layout in memory
def default_layouts(*shapes):
    return [range(len(shape) - 1, -1, -1) for shape in shapes]


# Register custom call targets
@functools.lru_cache
def _register_custom_calls():
    for _name, _value in dr._get_plugin(gl=True).registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="gpu")


# ================================================================================================
# Rasterize
# ================================================================================================

#### FORWARD ####


# @functools.partial(jax.jit, static_argnums=(0,))
def _rasterize_fwd_custom_call(r: "Renderer", pose, pos, tri, ranges, projMatrix, resolution):
    return _build_rasterize_fwd_primitive(r).bind(pose, pos, tri, ranges, projMatrix, resolution)


@functools.lru_cache(maxsize=None)
def _build_rasterize_fwd_primitive(r: "Renderer"):
    _register_custom_calls()
    # For JIT compilation we need a function to evaluate the shape and dtype of the
    # outputs of our op for some given inputs

    def _rasterize_fwd_abstract(pose, pos, tri, ranges, projection_matrix, resolution):
        if len(pos.shape) != 2 or pos.shape[-1] != 4:
            raise ValueError(
                "Pass in pos aa [num_vertices, 4] sized input"
            )
        # if len(pose.shape) != 3 or pose.shape[-1] != 4:
        #     raise ValueError(
        #         "Pass in pose aa [num_images, 4, 4] sized input"
        #     )
        num_images = pose.shape[0]

        dtype = dtypes.canonicalize_dtype(pose.dtype)

        return [
            ShapedArray(
                (num_images, r.intrinsics.height, r.intrinsics.width, 4), dtype
            ),
            ShapedArray(
                (num_images, r.intrinsics.height, r.intrinsics.width, 4), dtype
            ),
        ]

    # Provide an MLIR "lowering" of the rasterize primitive.
    def _rasterize_fwd_lowering(ctx, poses, pos, tri, ranges, projection_matrix, resolution):
        """
        Single-object (one obj represented by tri) rasterization with
        multiple poses (first dimension fo pos)
        dr.rasterize(glctx, pos, tri, resolution=resolution)
        """
        # Extract the numpy type of the inputs
        poses_aval, pos_aval, tri_aval, ranges_aval, projection_matrix_aval, resolution_aval = ctx.avals_in
        # if poses_aval.ndim != 3:
        #     raise NotImplementedError(
        #         f"Only 3D vtx position inputs supported: got {poses_aval.shape}"
        #     )
        # if tri_aval.ndim != 2:
        #     raise NotImplementedError(
        #         f"Only 2D triangle inputs supported: got {tri_aval.shape}"
        #     )
        # if resolution_aval.shape[0] != 2:
        #     raise NotImplementedError(
        #         f"Only 2D resolutions supported: got {resolution_aval.shape}"
        #     )

        np_dtype = np.dtype(poses_aval.dtype)
        if np_dtype != np.float32:
            raise NotImplementedError(f"Unsupported vtx positions dtype {np_dtype}")
        if np.dtype(tri_aval.dtype) != np.int32:
            raise NotImplementedError(f"Unsupported triangles dtype {tri_aval.dtype}")

        num_images = poses_aval.shape[0]
        num_objects = ranges_aval.shape[0]
        assert num_objects == poses_aval.shape[1], f"Number of poses {poses_aval.shape[1]} should match number of objects {num_objects}"
        num_vertices = pos_aval.shape[0]
        num_triangles = tri_aval.shape[0]
        out_shp_dtype = mlir.ir.RankedTensorType.get(
            [num_images, r.intrinsics.height, r.intrinsics.width, 4],
            mlir.dtype_to_ir_type(np_dtype),
        )

        opaque = dr._get_plugin(gl=True).build_diff_rasterize_fwd_descriptor(
            r.renderer_env.cpp_wrapper, [num_images, num_objects, num_vertices, num_triangles]
        )

        op_name = "jax_rasterize_fwd_gl"

        return custom_call(
            op_name,
            # Output types
            result_types=[out_shp_dtype, out_shp_dtype],
            # The inputs:
            operands=[poses, pos, tri, ranges, projection_matrix, resolution],
            backend_config=opaque,
            operand_layouts=[
                (3, 2, 0, 1),
                *default_layouts(
                    pos_aval.shape, tri_aval.shape, ranges_aval.shape, projection_matrix_aval.shape, resolution_aval.shape
                )
            ],
            result_layouts=default_layouts(
                (
                    num_images,
                    r.intrinsics.height,
                    r.intrinsics.width,
                    4,
                ),
                (
                    num_images,
                    r.intrinsics.height,
                    r.intrinsics.width,
                    4,
                ),
            ),
        ).results

    # *********************************************
    # *  REGISTER THE OP WITH JAX  *
    # *********************************************
    _rasterize_prim = core.Primitive(f"rasterize_multiple_fwd_{id(r)}")
    _rasterize_prim.multiple_results = True
    _rasterize_prim.def_impl(functools.partial(xla.apply_primitive, _rasterize_prim))
    _rasterize_prim.def_abstract_eval(_rasterize_fwd_abstract)

    # # Connect the XLA translation rules for JIT compilation
    mlir.register_lowering(_rasterize_prim, _rasterize_fwd_lowering, platform="gpu")

    return _rasterize_prim


#### BACKWARD ####


# @functools.partial(jax.jit, static_argnums=(0,))
def _rasterize_bwd_custom_call(r: "Renderer", pos, tri, rast_out, dy, ddb):
    return _build_rasterize_bwd_primitive(r).bind(pos, tri, rast_out, dy, ddb)


@functools.lru_cache(maxsize=None)
def _build_rasterize_bwd_primitive(r: "Renderer"):
    _register_custom_calls()
    # For JIT compilation we need a function to evaluate the shape and dtype of the
    # outputs of our op for some given inputs

    def _rasterize_bwd_abstract(pos, tri, rast_out, dy, ddb):
        if len(pos.shape) != 3:
            raise ValueError(
                "Pass in a [num_images, num_vertices, 4] sized first input"
            )
        out_shp = pos.shape
        dtype = dtypes.canonicalize_dtype(pos.dtype)

        return [ShapedArray(out_shp, dtype)]

    # Provide an MLIR "lowering" of the rasterize primitive.
    def _rasterize_bwd_lowering(ctx, pos, tri, rast_out, dy, ddb):
        # Extract the numpy type of the inputs
        pos_aval, tri_aval, rast_aval, dy_aval, ddb_aval = ctx.avals_in

        num_images, num_vertices = pos_aval.shape[:2]
        num_triangles = tri_aval.shape[0]
        depth, height, width = rast_aval.shape[:3]

        if rast_aval.ndim != 4:
            raise NotImplementedError(
                f"Rasterization output should be 4D: got {rast_aval.shape}"
            )
        if dy_aval.ndim != 4 or ddb_aval.ndim != 4:
            raise NotImplementedError(
                f"Grad outputs from rasterize should be 4D: got dy={dy_aval.shape} and ddb={ddb_aval.shape}"
            )

        np_dtype = np.dtype(rast_aval.dtype)
        if np_dtype != np.float32:
            raise NotImplementedError(f"Unsupported dtype {np_dtype}")

        out_shp_dtype = mlir.ir.RankedTensorType.get(
            [num_images, num_vertices, 4], mlir.dtype_to_ir_type(np_dtype)
        )  # gradients have same size as the positions

        opaque = dr._get_plugin(gl=True).build_diff_rasterize_bwd_descriptor(
            [num_images, num_vertices], [num_triangles], [depth, height, width]
        )

        op_name = "jax_rasterize_bwd"

        return custom_call(
            op_name,
            # Output types
            result_types=[out_shp_dtype],
            # The inputs:
            operands=[pos, tri, rast_out, dy, ddb],
            backend_config=opaque,
            operand_layouts=default_layouts(
                pos_aval.shape,
                tri_aval.shape,
                rast_aval.shape,
                dy_aval.shape,
                ddb_aval.shape,
            ),
            result_layouts=default_layouts(
                (
                    num_images,
                    num_vertices,
                    4,
                )
            ),
        ).results

    # *********************************************
    # *  REGISTER THE OP WITH JAX  *
    # *********************************************
    _rasterize_prim = core.Primitive(f"rasterize_multiple_bwd_{id(r)}")
    _rasterize_prim.multiple_results = True
    _rasterize_prim.def_impl(functools.partial(xla.apply_primitive, _rasterize_prim))
    _rasterize_prim.def_abstract_eval(_rasterize_bwd_abstract)

    # # Connect the XLA translation rules for JIT compilation
    mlir.register_lowering(_rasterize_prim, _rasterize_bwd_lowering, platform="gpu")

    return _rasterize_prim


# ================================================================================================
# Interpolate
# ================================================================================================

#### FORWARD ####


# @functools.partial(jax.jit, static_argnums=(0,))
def _interpolate_fwd_custom_call(
    r: "Renderer", attr, rast_out, tri, rast_db, diff_attrs_all, diff_attrs
):
    return _build_interpolate_fwd_primitive(r).bind(
        attr, rast_out, tri, rast_db, diff_attrs_all, diff_attrs
    )


# @functools.lru_cache(maxsize=None)
def _build_interpolate_fwd_primitive(r: "Renderer"):
    _register_custom_calls()
    # For JIT compilation we need a function to evaluate the shape and dtype of the
    # outputs of our op for some given inputs

    def _interpolate_fwd_abstract(
        attr, rast_out, tri, rast_db, diff_attrs_all, diff_attrs
    ):
        if len(attr.shape) != 3:
            raise ValueError(
                "Pass in a [num_images, num_vertices, num_attributes] sized first input"
            )
        num_images, num_vertices, num_attributes = attr.shape
        _, height, width, _ = rast_out.shape
        num_tri, _ = tri.shape
        num_diff_attrs = diff_attrs.shape[0]

        dtype = dtypes.canonicalize_dtype(attr.dtype)

        out_abstract = ShapedArray((num_images, height, width, num_attributes), dtype)
        out_db_abstract = ShapedArray(
            (num_images, height, width, 2 * num_diff_attrs), dtype
        )  # empty tensor
        return [out_abstract, out_db_abstract]

    # Provide an MLIR "lowering" of the interpolate primitive.
    def _interpolate_fwd_lowering(
        ctx, attr, rast_out, tri, rast_db, diff_attrs_all, diff_attrs
    ):
        # Extract the numpy type of the inputs
        (
            attr_aval,
            rast_out_aval,
            tri_aval,
            rast_db_aval,
            _,
            diff_attr_aval,
        ) = ctx.avals_in

        if attr_aval.ndim != 3:
            raise NotImplementedError(
                f"Only 3D attribute inputs supported: got {attr_aval.shape}"
            )
        if rast_out_aval.ndim != 4:
            raise NotImplementedError(
                f"Only 4D rast inputs supported: got {rast_out_aval.shape}"
            )
        if tri_aval.ndim != 2:
            raise NotImplementedError(
                f"Only 2D triangle tensors supported: got {tri_aval.shape}"
            )

        np_dtype = np.dtype(attr_aval.dtype)
        if np_dtype != np.float32:
            raise NotImplementedError(f"Unsupported attributes dtype {np_dtype}")
        if np.dtype(tri_aval.dtype) != np.int32:
            raise NotImplementedError(f"Unsupported triangle dtype {tri_aval.dtype}")
        if np.dtype(diff_attr_aval.dtype) != np.int32:
            raise NotImplementedError(
                f"Unsupported diff attribute dtype {diff_attr_aval.dtype}"
            )

        num_images, num_vertices, num_attributes = attr_aval.shape
        depth, height, width = rast_out_aval.shape[:3]
        num_triangles = tri_aval.shape[0]
        num_diff_attrs = diff_attr_aval.shape[0]

        if num_diff_attrs > 0 and rast_db_aval.shape[-1] < num_diff_attrs:
            raise NotImplementedError(
                f"Attempt to propagate bary gradients through {num_diff_attrs} attributes: got {rast_db_aval.shape}"
            )

        out_shp_dtype = mlir.ir.RankedTensorType.get(
            [num_images, height, width, num_attributes], mlir.dtype_to_ir_type(np_dtype)
        )
        out_db_shp_dtype = mlir.ir.RankedTensorType.get(
            [num_images, height, width, 2 * num_diff_attrs],
            mlir.dtype_to_ir_type(np_dtype),
        )

        opaque = dr._get_plugin(gl=True).build_diff_interpolate_descriptor(
            [num_images, num_vertices, num_attributes],
            [depth, height, width],
            [num_triangles],
            num_diff_attrs,  # diff wrt all attributes (TODO)
        )

        op_name = "jax_interpolate_fwd"

        return custom_call(
            op_name,
            # Output types
            result_types=[out_shp_dtype, out_db_shp_dtype],
            # The inputs:
            operands=[attr, rast_out, tri, rast_db, diff_attrs],
            backend_config=opaque,
            operand_layouts=default_layouts(
                attr_aval.shape,
                rast_out_aval.shape,
                tri_aval.shape,
                rast_db_aval.shape,
                diff_attr_aval.shape,
            ),
            result_layouts=default_layouts(
                (
                    num_images,
                    height,
                    width,
                    num_attributes,
                ),
                (
                    num_images,
                    height,
                    width,
                    num_attributes,
                ),
            ),
        ).results

    # *********************************************
    # *  REGISTER THE OP WITH JAX  *
    # *********************************************
    _interpolate_prim = core.Primitive(f"interpolate_multiple_fwd_{id(r)}")
    _interpolate_prim.multiple_results = True
    _interpolate_prim.def_impl(
        functools.partial(xla.apply_primitive, _interpolate_prim)
    )
    _interpolate_prim.def_abstract_eval(_interpolate_fwd_abstract)

    # # Connect the XLA translation rules for JIT compilation
    mlir.register_lowering(_interpolate_prim, _interpolate_fwd_lowering, platform="gpu")

    return _interpolate_prim


#### BACKWARD ####


# @functools.partial(jax.jit, static_argnums=(0,))
def _interpolate_bwd_custom_call(
    r: "Renderer",
    attr,
    rast_out,
    tri,
    dy,
    rast_db,
    dda,
    diff_attrs_all,
    diff_attrs_list,
):
    return _build_interpolate_bwd_primitive(r).bind(
        attr, rast_out, tri, dy, rast_db, dda, diff_attrs_all, diff_attrs_list
    )


# @functools.lru_cache(maxsize=None)
def _build_interpolate_bwd_primitive(r: "Renderer"):
    _register_custom_calls()
    # For JIT compilation we need a function to evaluate the shape and dtype of the
    # outputs of our op for some given inputs

    def _interpolate_bwd_abstract(
        attr, rast_out, tri, dy, rast_db, dda, diff_attrs_all, diff_attrs_list
    ):
        if len(attr.shape) != 3:
            raise ValueError(
                "Pass in a [num_images, num_vertices, num_attributes] sized first input"
            )
        num_images, num_vertices, num_attributes = attr.shape
        depth, height, width, rast_channels = rast_out.shape
        depth_db, height_db, width_db, rast_channels_db = rast_db.shape

        dtype = dtypes.canonicalize_dtype(attr.dtype)

        g_attr_abstract = ShapedArray((num_images, num_vertices, num_attributes), dtype)
        g_rast_abstract = ShapedArray((depth, height, width, rast_channels), dtype)
        g_rast_db_abstract = ShapedArray(
            (depth_db, height_db, width_db, rast_channels_db), dtype
        )
        return [g_attr_abstract, g_rast_abstract, g_rast_db_abstract]

    # Provide an MLIR "lowering" of the interpolate primitive.
    def _interpolate_bwd_lowering(
        ctx, attr, rast_out, tri, dy, rast_db, dda, diff_attrs_all, diff_attrs_list
    ):
        # Extract the numpy type of the inputs
        (
            attr_aval,
            rast_out_aval,
            tri_aval,
            dy_aval,
            rast_db_aval,
            dda_aval,
            _,
            diff_attr_aval,
        ) = ctx.avals_in

        if attr_aval.ndim != 3:
            raise NotImplementedError(
                f"Only 3D attribute inputs supported: got {attr_aval.shape}"
            )
        if rast_out_aval.ndim != 4:
            raise NotImplementedError(
                f"Only 4D rast inputs supported: got {rast_out_aval.shape}"
            )
        if tri_aval.ndim != 2:
            raise NotImplementedError(
                f"Only 2D triangle tensors supported: got {tri_aval.shape}"
            )

        np_dtype = np.dtype(attr_aval.dtype)
        if np_dtype != np.float32:
            raise NotImplementedError(f"Unsupported attributes dtype {np_dtype}")
        if np.dtype(tri_aval.dtype) != np.int32:
            raise NotImplementedError(f"Unsupported triangle dtype {tri_aval.dtype}")

        num_images, num_vertices, num_attributes = attr_aval.shape
        depth, height, width, rast_channels = rast_out_aval.shape
        depth_db, height_db, width_db, rast_channels_db = rast_db_aval.shape
        num_triangles = tri_aval.shape[0]
        num_diff_attrs = diff_attr_aval.shape[0]

        g_attr_shp_dtype = mlir.ir.RankedTensorType.get(
            [num_images, num_vertices, num_attributes], mlir.dtype_to_ir_type(np_dtype)
        )
        g_rast_shp_dtype = mlir.ir.RankedTensorType.get(
            [depth, height, width, rast_channels], mlir.dtype_to_ir_type(np_dtype)
        )
        g_rast_db_shp_dtype = mlir.ir.RankedTensorType.get(
            [depth_db, height_db, width_db, rast_channels_db],
            mlir.dtype_to_ir_type(np_dtype),
        )

        opaque = dr._get_plugin(gl=True).build_diff_interpolate_descriptor(
            [num_images, num_vertices, num_attributes],
            [depth, height, width],
            [num_triangles],
            num_diff_attrs,
        )

        op_name = "jax_interpolate_bwd"

        return custom_call(
            op_name,
            # Output types
            result_types=[g_attr_shp_dtype, g_rast_shp_dtype, g_rast_db_shp_dtype],
            # The inputs:
            operands=[attr, rast_out, tri, dy, rast_db, dda, diff_attrs_list],
            backend_config=opaque,
            operand_layouts=default_layouts(
                attr_aval.shape,
                rast_out_aval.shape,
                tri_aval.shape,
                dy_aval.shape,
                rast_db_aval.shape,
                dda_aval.shape,
                diff_attr_aval.shape,
            ),
            result_layouts=default_layouts(
                (
                    num_images,
                    num_vertices,
                    num_attributes,
                ),
                (
                    depth,
                    height,
                    width,
                    rast_channels,
                ),
                (
                    depth_db,
                    height_db,
                    width_db,
                    rast_channels_db,
                ),
            ),
        ).results

    # *********************************************
    # *  REGISTER THE OP WITH JAX  *
    # *********************************************
    _interpolate_prim = core.Primitive(f"interpolate_multiple_bwd_{id(r)}")
    _interpolate_prim.multiple_results = True
    _interpolate_prim.def_impl(
        functools.partial(xla.apply_primitive, _interpolate_prim)
    )
    _interpolate_prim.def_abstract_eval(_interpolate_bwd_abstract)

    # # Connect the XLA translation rules for JIT compilation
    mlir.register_lowering(_interpolate_prim, _interpolate_bwd_lowering, platform="gpu")

    return _interpolate_prim
