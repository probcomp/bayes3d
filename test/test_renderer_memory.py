import numpy as np
import jax.numpy as jnp
from scipy.spatial.transform import Rotation as R
import jax
import bayes3d as b
import trimesh
import os
import gc
import time
import GPUtil

# gpu = GPUtil.getGPUs()[0].memoryUsed

avail_gpumem_preinit = GPUtil.getGPUs()[0].memoryUsed
print(avail_gpumem_preinit)


intrinsics = b.Intrinsics(
    300,
    300,
    200.0,200.0,
    150.0,150.0,
    0.001, 50.0
)
b.setup_renderer(intrinsics, num_layers=1)
renderer = b.RENDERER

avail_gpumem_postinit = GPUtil.getGPUs()[0].memoryUsed
print(avail_gpumem_postinit)

for i in range(1):
    renderer.add_mesh_from_file(os.path.join(b.utils.get_assets_dir(),"sample_objs/cube.obj")\
                                , mesh_name = "cube_{}".format(i))

avail_gpumem_postmesh = GPUtil.getGPUs()[0].memoryUsed
print(avail_gpumem_postmesh)

poses = [b.t3d.transform_from_pos(jnp.array([-3.0, -3.0, 4.0]))]
delta_pose = b.t3d.transform_from_rot_and_pos(
    R.from_euler('zyx', [1.0, -0.1, -2.0], degrees=True).as_matrix(),
    jnp.array([0.09, 0.05, 0.02])
)
for t in range(5):
    poses.append(poses[-1].dot(delta_pose))
poses = jnp.stack(poses)
images_1 = b.RENDERER.render_many(poses[:,None,...],  jnp.array([0]))
print(images_1.shape)


avail_gpumem_postrender = GPUtil.getGPUs()[0].memoryUsed
print(avail_gpumem_postrender)

renderer.clear_gpu_mem()
# renderer.renderer_env.release_context()
avail_gpumem_postreset = GPUtil.getGPUs()[0].memoryUsed

print(avail_gpumem_postreset)

backend = jax.lib.xla_bridge.get_backend()
for buf in backend.live_buffers(): 
    buf.delete()

del renderer
del b.RENDERER
gc.collect()
    
print(GPUtil.getGPUs()[0].memoryUsed)
