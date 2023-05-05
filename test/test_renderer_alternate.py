import numpy as np
import jax.numpy as jnp
import jax
import bayes3d as b
import trimesh
import os
import time
import bayes3d.nvdiffrast.common as dr
import torch

intrinsics = b.Intrinsics(
    300,
    300,
    200.0,200.0,
    150.0,150.0,
    0.001, 50.0
)

b.setup(intrinsics.height, intrinsics.width, 1024)

mesh  = trimesh.load(os.path.join(b.utils.get_assets_dir(),"sample_objs/cube.obj"))
vertices = np.array(mesh.vertices)
vertices = np.concatenate([vertices, np.ones((*vertices.shape[:-1],1))],axis=-1)
triangles = np.array(mesh.faces)

mod.load_vertices_fwd(
    torch.tensor(vertices.astype("f"), device='cuda'),
    torch.tensor(triangles.astype(np.int32), device='cuda'),
)


gt_pose = jnp.array([
    [1.0, 0.0, 0.0, 0.0],   
    [0.0, 1.0, 0.0, -0.0],   
    [0.0, 0.0, 1.0, 3.0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)
gt_pose = b.distributions.gaussian_vmf_sample(jax.random.PRNGKey(10), gt_pose, 0.00001, 0.0001)
poses = jnp.tile(gt_pose[None,None,...],(1,10,1,1))

proj_list = list(b.camera.open_gl_projection_matrix(
    intrinsics.height, intrinsics.width, intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy, intrinsics.near, intrinsics.far
).reshape(-1))



image = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(

    mod.rasterize_fwd_gl(
    
        torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(poses))
, 

proj_list, [0], 0))
)
b.get_depth_image(image[5,:,:,2]).save("test_renderer.png")


image = jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(

    mod.rasterize_fwd_gl_jax(
    
        torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(poses))
    )))
b.get_depth_image(image[5,:,:,2]).save("test_renderer.png")







# renderer.add_mesh_from_file(os.path.join(j.utils.get_assets_dir(),"sample_objs/sphere.obj"))



from IPython import embed; embed()

# renderer = j.Renderer(intrinsics)

# r = 0.1
# outlier_prob = 0.01

# max_depth = 15.0

# renderer.add_mesh_from_file(os.path.join(j.utils.get_assets_dir(),"sample_objs/cube.obj"))
# renderer.add_mesh_from_file(os.path.join(j.utils.get_assets_dir(),"sample_objs/sphere.obj"))

# num_parallel_frames = 20
# gt_poses_1 = jnp.tile(jnp.array([
#     [1.0, 0.0, 0.0, 0.0],   
#     [0.0, 1.0, 0.0, -1.0],   
#     [0.0, 0.0, 1.0, 8.0],   
#     [0.0, 0.0, 0.0, 1.0],   
#     ]
# )[None,...],(num_parallel_frames,1,1))
# gt_poses_1 = gt_poses_1.at[:,0,3].set(jnp.linspace(-2.0, 2.0, gt_poses_1.shape[0]))
# gt_poses_1 = gt_poses_1.at[:,2,3].set(jnp.linspace(10.0, 5.0, gt_poses_1.shape[0]))

# gt_poses_2 = jnp.tile(jnp.array([
#     [1.0, 0.0, 0.0, 0.0],   
#     [0.0, 1.0, 0.0, -1.0],   
#     [0.0, 0.0, 1.0, 8.0],   
#     [0.0, 0.0, 0.0, 1.0],   
#     ]
# )[None,...],(num_parallel_frames,1,1))
# gt_poses_2 = gt_poses_2.at[:,0,3].set(jnp.linspace(4.0, -3.0, gt_poses_2.shape[0]))
# gt_poses_2 = gt_poses_2.at[:,2,3].set(jnp.linspace(12.0, 5.0, gt_poses_2.shape[0]))

# gt_poses_all = jnp.stack([gt_poses_1, gt_poses_2])

# single_image = renderer.render_single_object(gt_poses_all[0,0], 0)
# single_image_viz = j.get_depth_image(single_image[:,:,2], max=max_depth)

# parallel_images = renderer.render_parallel(gt_poses_all[0,:], 0)
# parallel_images_viz = j.get_depth_image(parallel_images[0,:,:,2], max=max_depth)

# multiobject_scene_img = renderer.render_multiobject(gt_poses_all[:,-1,:,:], [0, 1])
# multiobject_viz = j.get_depth_image(multiobject_scene_img[:,:,2], max=max_depth)

# multiobject_scene_parallel_img = renderer.render_multiobject_parallel(gt_poses_all[:,:,:,:], [0, 1])
# multiobject_parallel_viz = j.get_depth_image(multiobject_scene_parallel_img[-1,::,:,2], max=max_depth)

# segmentation_viz = j.get_depth_image(multiobject_scene_parallel_img[-1,:,:,3], max=4.0)

# j.multi_panel(
#     [single_image_viz, parallel_images_viz, multiobject_viz, multiobject_parallel_viz, segmentation_viz]
# ).save("test_renderer.png")

# from IPython import embed; embed()

#############
# JAX layer #
#############

from functools import partial, reduce

from jax import core, dtypes
from jax.interpreters import xla
from jax.interpreters import mlir
from jax.interpreters.mlir import ir
from jaxlib.hlo_helpers import custom_call
from jax.lib import xla_client
from jax.abstract_arrays import ShapedArray

_rasterize_fwd_p = core.Primitive("rasterize")
_rasterize_fwd_p.multiple_results = True
_rasterize_fwd_p.def_impl(partial(xla.apply_primitive), _rasterize_fwd_p)

# TODO: determine what goes in to args (actually).
def rasterize_fwd(arr):
    return _rasterize_fwd_p.bind(arr)

#####
# Lowering
#####

# Register functions defined in mod as custom call target for GPUs.
for _name, _value in mod.get_rasterize_registrations().items():
    xla_client.register_custom_call_target(_name, _value, platform="gpu")

# TODO: this actually calls out to the CUDA pybind'd stuff.
def _rasterize_fwd_cuda_lowering(ctx, arr):
    # TODO:
    out = custom_call(
        b"rasterize_fwd", # name of call in PyBind module.
        out_types=[], # TODO.
        operands=[], # TODO.
        backend_config=None, # TODO.
        operand_layouts=None, # TODO.
        result_layouts=None, # TODO.
    )
    return out 

mlir.register_lowering(_rasterize_fwd_p, _rasterize_fwd_cuda_lowering, platform="gpu")

#####
# Abstract evaluation (tracing)
#####

# (M, N, 4, 4) --> (N, 100, 100, 4)
def _rasterize_fwd_abstract(arr):
    arr_dtype = dtypes.canonicalize_dtype(arr.dtype)
    (M, N) = arr.shape[0], arr.shape[1]
    return (ShapedArray(
        (N, 100, 100, 4), 
        arr_dtype, 
        named_shape = arr.named_shape), )

_rasterize_fwd_p.def_abstract_eval(_rasterize_fwd_abstract)