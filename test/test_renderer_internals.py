import numpy as np
import jax.numpy as jnp
import jax
import bayes3d as b
import trimesh
import os
import time
import torch
import bayes3d._rendering.nvdiffrast.common as dr


intrinsics = b.Intrinsics(
    300,
    300,
    200.0,200.0,
    150.0,150.0,
    0.001, 50.0
)
b.setup_renderer(intrinsics)
renderer = b.RENDERER

r = 0.1
outlier_prob = 0.01
max_depth = 15.0

renderer.add_mesh_from_file(os.path.join(b.utils.get_assets_dir(),"sample_objs/cube.obj"))
renderer.add_mesh_from_file(os.path.join(b.utils.get_assets_dir(),"sample_objs/sphere.obj"))



poses = jnp.array([
    [1.0, 0.0, 0.0, 0.0],   
    [0.0, 1.0, 0.0, -1.0],   
    [0.0, 0.0, 1.0, 8.0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)[None, None, ...]
indices = jnp.array([0])

img = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(dr._get_plugin(gl=True).rasterize_fwd_gl(
    b.RENDERER.renderer_env.cpp_wrapper, 
    torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(jnp.tile(poses, (1,2,1,1)))),
    b.RENDERER.proj_list,
    [0]
)))
b.get_depth_image(img[0,:,:,2]).save("1.png")
assert not jnp.all(img[0,:,:,2] == 0.0)

multiobject_scene_img = renderer._render_many(poses, jnp.array([0]))[0]
b.get_depth_image(multiobject_scene_img[:,:,2]).save("0.png")
assert not jnp.all( multiobject_scene_img[:,:,2] == 0.0)



# from jax.interpreters import batching, mlir, xla
# from jax.lib import xla_client
# from jax.core import ShapedArray
# from jax.interpreters import ad, batching, mlir, xla
# from jax.lib import xla_client
# from jaxlib.hlo_helpers import custom_call
# from jax import core, dtypes
# import functools

# r = b.RENDERER

# for _name, _value in dr._get_plugin(gl=True).registrations().items():
#     print(_name, _value)
#     xla_client.register_custom_call_target(_name, _value, platform="gpu")

# def _render_abstract(poses, indices):
#     num_images = poses.shape[0]
#     if poses.shape[1] != indices.shape[0]:
#         raise ValueError(f"Mismatched #objects: {poses.shape}, {indices.shape}")
#     dtype = dtypes.canonicalize_dtype(poses.dtype)
#     return [ShapedArray((num_images, r.intrinsics.height, r.intrinsics.width, 4), dtype),
#             ShapedArray((), dtype)]

# def _render_lowering(ctx, poses, indices):
#     # Extract the numpy type ofthe inputs
#     poses_aval, indices_aval = ctx.avals_in

#     num_images, num_objects = poses_aval.shape[:2]
#     out_shp_dtype = mlir.ir.RankedTensorType.get(
#         [num_images, r.intrinsics.height, r.intrinsics.width, 4],
#         mlir.dtype_to_ir_type(poses_aval.dtype))

#     if num_objects != indices_aval.shape[0]:
#         raise ValueError("Mismatched #objects in poses vs indices: "
#                             f"{num_objects} vs {indices_aval.shape[0]}")
#     opaque = dr._get_plugin(gl=True).build_rasterize_descriptor(r.renderer_env.cpp_wrapper,
#                                                                 r.proj_list,
#                                                                 [num_objects, num_images])

#     scalar_dummy = mlir.ir.RankedTensorType.get([], mlir.dtype_to_ir_type(poses_aval.dtype))
#     op_name = "jax_rasterize_fwd_gl"
#     return custom_call(
#         op_name,
#         # Output types
#         out_types=[out_shp_dtype, scalar_dummy],
#         # The inputs:
#         operands=[poses, indices],
#         # Layout specification:
#         operand_layouts=[(3, 2, 1, 0), (0,)],
#         result_layouts=[(3, 2, 1, 0), ()],
#         # GPU specific additional data
#         backend_config=opaque
#     )



# # *********************************************
# # *  BOILERPLATE TO REGISTER THE OP WITH JAX  *
# # *********************************************
# _render_prim = core.Primitive(f"render_multiple_{id(r)}")
# _render_prim.multiple_results = True
# _render_prim.def_impl(functools.partial(xla.apply_primitive, _render_prim))
# _render_prim.def_abstract_eval(_render_abstract)
# # Connect the XLA translation rules for JIT compilation
# mlir.register_lowering(_render_prim, _render_lowering, platform="gpu")

# out_img = _render_prim.bind(poses, indices)

# assert not jnp.all(out_img[:,:,2] == 0.0)
