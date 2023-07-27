import numpy as np
import jax.numpy as jnp
import jax
import bayes3d as b
import trimesh
import os
import time


intrinsics = b.Intrinsics(
    300,
    300,
    200.0,200.0,
    150.0,150.0,
    0.001, 50.0
)
b.setup_renderer(intrinsics, num_layers=1)
renderer = b.RENDERER

r = 0.1
outlier_prob = 0.01
max_depth = 15.0

renderer.add_mesh_from_file(os.path.join(b.utils.get_assets_dir(),"sample_objs/cube.obj"))
renderer.add_mesh_from_file(os.path.join(b.utils.get_assets_dir(),"sample_objs/sphere.obj"))



num_parallel_frames = 20
gt_poses_1 = jnp.tile(jnp.array([
    [1.0, 0.0, 0.0, 0.0],   
    [0.0, 1.0, 0.0, -1.0],   
    [0.0, 0.0, 1.0, 8.0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)[None,...],(num_parallel_frames,1,1))
gt_poses_1 = gt_poses_1.at[:,0,3].set(jnp.linspace(-2.0, 2.0, gt_poses_1.shape[0]))
gt_poses_1 = gt_poses_1.at[:,2,3].set(jnp.linspace(10.0, 5.0, gt_poses_1.shape[0]))

gt_poses_2 = jnp.tile(jnp.array([
    [1.0, 0.0, 0.0, 0.0],   
    [0.0, 1.0, 0.0, -1.0],   
    [0.0, 0.0, 1.0, 8.0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)[None,...],(num_parallel_frames,1,1))
gt_poses_2 = gt_poses_2.at[:,0,3].set(jnp.linspace(4.0, -3.0, gt_poses_2.shape[0]))
gt_poses_2 = gt_poses_2.at[:,2,3].set(jnp.linspace(12.0, 5.0, gt_poses_2.shape[0]))

gt_poses_all = jnp.stack([gt_poses_1, gt_poses_2],axis=1)

indices = jnp.array( [0, 1])


multiobject_scene_img = renderer.render(gt_poses_all[-1, ...], jnp.array([0, 1]))
multiobject_viz = b.get_depth_image(multiobject_scene_img[:,:,2], max=max_depth)

multiobject_scene_parallel_img = renderer.render_many(gt_poses_all, jnp.array([0, 1]))
multiobject_parallel_viz = b.get_depth_image(multiobject_scene_parallel_img[-1,:,:,2], max=max_depth)

segmentation_viz = b.get_depth_image(multiobject_scene_parallel_img[-1,:,:,3], max=4.0)

images = [b.get_depth_image(multiobject_scene_parallel_img[i,:,:,2], max=max_depth) for i in range(num_parallel_frames)]
b.multi_panel(
    [multiobject_viz, multiobject_parallel_viz, segmentation_viz] + images
).save("test_renderer.png")


def test_segmentation_produces_sensical_outputs():
    assert jnp.allclose(multiobject_scene_parallel_img[-1,:,:,3].max(), 2.0)
    assert jnp.allclose(multiobject_scene_parallel_img[-1,:,:,3].min(), 0.0)
    assert jnp.allclose(multiobject_scene_img[:,:,3].max(), 2.0)
    assert jnp.allclose(multiobject_scene_img[:,:,3].min(), 0.0)

def test_something_is_being_rendered():
    assert not jnp.all(multiobject_scene_parallel_img[0,:,:,2] == intrinsics.far)
    assert not jnp.all(multiobject_scene_parallel_img[-1,:,:,2] == intrinsics.far)
    assert not jnp.all(multiobject_scene_img[:,:,2] == intrinsics.far)
