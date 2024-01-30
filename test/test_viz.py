import os

import bayes3d as b
import jax.numpy as jnp
import matplotlib.pyplot as plt

bop_ycb_dir = os.path.join(b.utils.get_assets_dir(), "bop/ycbv")
rgbd, gt_ids, gt_poses, masks = b.utils.ycb_loader.get_test_img("52", "1", bop_ycb_dir)
fig = b.viz_depth_image(rgbd.depth)
fig.savefig("depth.png", **b.saveargs)
fig = b.viz_rgb_image(rgbd.rgb)
fig.savefig("rgb.png", **b.saveargs)

fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
b.add_rgb_image(ax, rgbd.rgb)
ax.set_title("RGB")
ax = fig.add_subplot(1, 2, 2)
b.add_depth_image(ax, rgbd.depth)
ax.set_title("DEPTH")
fig.savefig("fig.png", **b.saveargs)


##################################################################################
#  Testing 2 edge cases in getting color-mapped depth image from rendered output #
##################################################################################

# set up renderer
intrinsics = b.Intrinsics(50, 50, 200.0, 200.0, 25.0, 25.0, 0.001, 20.0)
b.setup_renderer(intrinsics)
renderer = b.RENDERER
renderer.add_mesh_from_file(
    os.path.join(b.utils.get_assets_dir(), "sample_objs/cube.obj")
)


# Test 1: check if b.get_depth_image returns a valid image if there is no object in the scene
no_object_in_scene_pose = jnp.array(
    [
        [1.0, 0.0, 0.0, -100.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 10.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
depth = renderer.render(no_object_in_scene_pose[None, ...], jnp.array([0]))[..., 2]
depth_image = b.scale_image(b.get_depth_image(depth), 8)
depth_image.save("viz_test_no_object_in_scene.png")

# Test 2: check if b.get_depth_image returns a valid image if object has only one unique depth value
object_unique_depth_pose = jnp.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 10.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
depth = renderer.render(object_unique_depth_pose[None, ...], jnp.array([0]))[..., 2]
assert jnp.unique(depth).size == 2  # far and object's depth
depth_image = b.scale_image(b.get_depth_image(depth), 8)
depth_image.save("viz_test_object_unique_depth.png")
