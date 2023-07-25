from typing import Tuple

import functools
import bayes3d._rendering.nvdiffrast.common as dr
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
from jaxlib.hlo_helpers import custom_call
from tqdm import tqdm

def _transform_image_zeros(image_jnp, intrinsics):
    image_jnp_2 = jnp.concatenate(
        [
            j.t3d.unproject_depth(image_jnp[:,:,2], intrinsics),
            image_jnp[:,:,3:]
        ],
        axis=-1
    )
    return image_jnp_2
_transform_image_zeros_jit = jax.jit(_transform_image_zeros)
_transform_image_zeros_parallel = jax.vmap(_transform_image_zeros, in_axes=(0,None))
_transform_image_zeros_parallel_jit = jax.jit(_transform_image_zeros_parallel)

def setup_renderer(intrinsics, num_layers=1024):
    """Setup the renderer.
    Args:
        intrinsics (bayes3d.camera.Intrinsics): The camera intrinsics.    
    """
    b.RENDERER = Renderer(intrinsics, num_layers=num_layers)

class Renderer(object):
    def __init__(self, intrinsics, num_layers=1024):
        """A renderer for rendering meshes.
        
        Args:
            intrinsics (bayes3d.camera.Intrinsics): The camera intrinsics.
            num_layers (int, optional): The number of scenes to render in parallel. Defaults to 1024.
        """
        self.intrinsics = intrinsics
        self.renderer_env = dr.RasterizeGLContext(intrinsics.height, intrinsics.width, output_db=False)
        self.proj_list = list(bayes3d.camera._open_gl_projection_matrix(
            intrinsics.height, intrinsics.width, 
            intrinsics.fx, intrinsics.fy, 
            intrinsics.cx, intrinsics.cy, 
            intrinsics.near, intrinsics.far
        ).reshape(-1))

        build_setup_primitive(self,intrinsics.height, intrinsics.width, num_layers).bind()


        self.meshes =[]
        self.mesh_names =[]
        self.model_box_dims = jnp.zeros((0,3))

    def add_mesh_from_file(self, mesh_filename, mesh_name=None, scaling_factor=1.0, force=None, center_mesh=True):
        """Add a mesh to the renderer from a file.
        
        Args:
            mesh_filename (str): The filename of the mesh.
            mesh_name (str, optional): The name of the mesh. Defaults to None.
            scaling_factor (float, optional): The scaling factor to apply to the mesh. Defaults to 1.0.
            force (str, optional): The file format to force. Defaults to None.
            center_mesh (bool, optional): Whether to center the mesh. Defaults to True.
        """
        mesh = trimesh.load(mesh_filename, force=force)
        self.add_mesh(mesh, mesh_name=mesh_name, scaling_factor=scaling_factor, center_mesh=center_mesh)

    def add_mesh(self, mesh, mesh_name=None, scaling_factor=1.0, center_mesh=True):
        """Add a mesh to the renderer.
        
        Args:
            mesh (trimesh.Trimesh): The mesh to add.
            mesh_name (str, optional): The name of the mesh. Defaults to None.
            scaling_factor (float, optional): The scaling factor to apply to the mesh. Defaults to 1.0.
            center_mesh (bool, optional): Whether to center the mesh. Defaults to True.
        """
        if mesh_name is None:
            mesh_name = f"object_{len(self.meshes)}"
        
        mesh.vertices = mesh.vertices * scaling_factor
        
        bounding_box_dims, bounding_box_pose = bayes3d.utils.aabb(mesh.vertices)
        if center_mesh:
            if not jnp.isclose(bounding_box_pose[:3,3], 0.0).all():
                print(f"Centering mesh with translation {bounding_box_pose[:3,3]}")
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
        prim = build_load_vertices_primitive(self)
        prim.bind(jnp.float32(vertices), jnp.int32(triangles))

    def _render_many(self, poses, indices):
        return _render_custom_call(self, poses, indices)[0]

    def render_many(self, poses, indices):
        """Render many scenes in parallel.
        
        Args:
            poses (jnp.ndarray): The poses of the objects in the scene. Shape (N, M, 4, 4)
                                 where N is the number of scenes and M is the number of objects.
                                 and the last two dimensions are the 4x4 poses.
            indices (jnp.ndarray): The indices of the objects to render. Shape (M,)

        Outputs:
            jnp.ndarray: The rendered images. Shape (N, H, W, 4) where N is the number of scenes
                         the final dimension is the segmentation image.
        """
        images_jnp = _render_custom_call(self, poses, indices)[0]
        return _transform_image_zeros_parallel(images_jnp, self.intrinsics)

    def render(self, poses, indices):
        """Render a single scene.

        Args:
            poses (jnp.ndarray): The poses of the objects in the scene. Shape (M, 4, 4)
            indices (jnp.ndarray): The indices of the objects to render. Shape (M,)
        Outputs:
            jnp.ndarray: The rendered image. Shape (H, W, 4)
                         the final dimension is the segmentation image.

        """
        return self.render_many(poses[None,...], indices)[0]


# Useful reference for understanding the custom calls setup:
#   https://github.com/dfm/extending-jax

@functools.lru_cache
def _register_custom_calls():
    for _name, _value in dr._get_plugin(gl=True).registrations().items():
        xla_client.register_custom_call_target(_name, _value, platform="gpu")

@functools.partial(jax.jit, static_argnums=(0,))
def _render_custom_call(r: "Renderer", poses, idx):
    return _build_render_primitive(r).bind(poses, idx)


@functools.lru_cache(maxsize=None)
def _build_render_primitive(r: "Renderer"):
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
        renders, dummy = _render_custom_call(r, poses, indices)

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




@functools.lru_cache(maxsize=None)
def build_setup_primitive(r: "Renderer", h, w, num_layers):
    _register_custom_calls()
    # print('build_setup_primitive')
    
    # For JIT compilation we need a function to evaluate the shape and dtype of the
    # outputs of our op for some given inputs
    def _setup_abstract():
        # print('setup abstract eval')
        dtype = dtypes.canonicalize_dtype(np.float32)
        return [ShapedArray((), dtype), ShapedArray((), dtype)]

    # Provide an MLIR "lowering" of the load_vertices primitive.
    def _setup_lowering(ctx):
        # print('lowering setup!')

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
    # print('build_load_vertices_primitive')
    
    # For JIT compilation we need a function to evaluate the shape and dtype of the
    # outputs of our op for some given inputs
    def _load_vertices_abstract(vertices, triangles):
        # print('load_vertices abstract eval:', vertices, triangles)
        dtype = dtypes.canonicalize_dtype(np.float32)
        return [ShapedArray((), dtype), ShapedArray((), dtype)]

    # Provide an MLIR "lowering" of the load_vertices primitive.
    def _load_vertices_lowering(ctx, vertices, triangles):
        # print('lowering load_vertices!')
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


