import numpy as np
import jax.numpy as jnp
import jax
import jax3dp3
import trimesh
import os

h, w, fx,fy, cx,cy = (
    300,
    300,
    200.0,200.0,
    150.0,150.0
)
near,far = 0.001, 50.0
r = 0.1
outlier_prob = 0.01
jax3dp3.setup_renderer(h, w, fx, fy, cx, cy, near, far)
mesh = trimesh.load(os.path.join(jax3dp3.utils.get_assets_dir(),"cube.obj"))
jax3dp3.load_model(mesh, h,w)

gt_poses = jnp.tile(jnp.array([
    [1.0, 0.0, 0.0, 0.0],   
    [0.0, 1.0, 0.0, -1.0],   
    [0.0, 0.0, 1.0, 8.0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)[None,...],(10,1,1))
gt_poses = gt_poses.at[:,0,3].set(jnp.linspace(-5.0, 5.0, gt_poses.shape[0]))

multiobject_scene_img = jax3dp3.render_multiobject(gt_poses, h,w, [0 for _ in range(gt_poses.shape[0])])
jax3dp3.viz.save_depth_image(multiobject_scene_img[:,:,2], "gt_image.png", max=10.0)



parallel_single_object_img = jax3dp3.render_multiobject_parallel(gt_poses[:,None, :,:], h,w, [0])
jax3dp3.viz.save_depth_image(parallel_single_object_img[0,:,:,2], "img_1.png", max=10.0)
jax3dp3.viz.save_depth_image(parallel_single_object_img[-1,:,:,2], "img_2.png", max=10.0)

from IPython import embed; embed()