import numpy as np
import jax.numpy as jnp
import jax
import jax3dp3
import trimesh
import os
import time

h, w, fx,fy, cx,cy = (
    300,
    300,
    200.0,200.0,
    150.0,150.0
)
near,far = 0.001, 50.0
r = 0.1
outlier_prob = 0.01

max_depth = 15.0

num_layers = 10
jax3dp3.setup_renderer(h, w, fx, fy, cx, cy, near, far,num_layers=num_layers)
mesh1 = trimesh.load(os.path.join(jax3dp3.utils.get_assets_dir(),"sample_objs/cube.obj"))
mesh2 = trimesh.load(os.path.join(jax3dp3.utils.get_assets_dir(),"sample_objs/sphere.obj"))
jax3dp3.load_model(mesh1)
jax3dp3.load_model(mesh2)

gt_poses = jnp.tile(jnp.array([
    [1.0, 0.0, 0.0, 0.0],   
    [0.0, 1.0, 0.0, -1.0],   
    [0.0, 0.0, 1.0, 8.0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)[None,...],(num_layers,1,1))
gt_poses = gt_poses.at[:,0,3].set(jnp.linspace(-2.0, 2.0, gt_poses.shape[0]))
gt_poses = gt_poses.at[:,2,3].set(jnp.linspace(10.0, 5.0, gt_poses.shape[0]))

parallel_images = jax3dp3.render_parallel(gt_poses, 0)

jax3dp3.utils.time_code_block(jax3dp3.render_parallel, (gt_poses, 1))
jax3dp3.utils.time_code_block(jax3dp3.render_parallel, (gt_poses, 1))

jax3dp3.viz.save_depth_image(parallel_images[0,:,:,2], "img_1.png", max=max_depth)
jax3dp3.viz.save_depth_image(parallel_images[-1,:,:,2], "img_2.png", max=max_depth)





gt_poses = jnp.tile(jnp.array([
    [1.0, 0.0, 0.0, 0.0],   
    [0.0, 1.0, 0.0, -1.0],   
    [0.0, 0.0, 1.0, 8.0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)[None,...],(5,1,1))
gt_poses = gt_poses.at[:,0,3].set(jnp.linspace(-2.0, 2.0, gt_poses.shape[0]))
gt_poses = gt_poses.at[:,2,3].set(jnp.linspace(10.0, 5.0, gt_poses.shape[0]))

gt_poses_2 = jnp.tile(jnp.array([
    [1.0, 0.0, 0.0, 0.0],   
    [0.0, 1.0, 0.0, -1.0],   
    [0.0, 0.0, 1.0, 8.0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)[None,...],(5,1,1))
gt_poses_2 = gt_poses_2.at[:,0,3].set(jnp.linspace(4.0, -3.0, gt_poses_2.shape[0]))
gt_poses_2 = gt_poses_2.at[:,2,3].set(jnp.linspace(12.0, 5.0, gt_poses_2.shape[0]))

from IPython import embed; embed()

gt_poses_full = jnp.stack([gt_poses, gt_poses_2])

images_independent = []
for i in range(gt_poses_full.shape[1]):
    multiobject_scene_img = jax3dp3.render_multiobject(gt_poses_full[:,i,...], [0, 1])
    images_independent.append(
        jax3dp3.viz.get_depth_image(multiobject_scene_img[:,:,2], max=max_depth)
    )
jax3dp3.viz.multi_panel(images_independent).save("independent.png")

images = []
multiobject_scene_img = jax3dp3.render_multiobject_parallel(gt_poses_full, [0, 1])
for i in range(multiobject_scene_img.shape[0]):
    images.append(
        jax3dp3.viz.get_depth_image(multiobject_scene_img[i,:,:,2], max=max_depth)
    )
jax3dp3.viz.multi_panel(images).save("parallel.png")

jax3dp3.viz.save_depth_image(multiobject_scene_img[4,:,:,2], "gt_image2.png", max=max_depth)



parallel_single_object_img = jax3dp3.render_multiobject_parallel(gt_poses_full,  [0,1])
jax3dp3.viz.save_depth_image(parallel_single_object_img[0,:,:,2], "parallel_single_img_1.png", max=max_depth)
jax3dp3.viz.save_depth_image(parallel_single_object_img[1,:,:,2], "parallel_single_img_2.png", max=max_depth)

from IPython import embed; embed()

