import jax.numpy as jnp
import bayes3d as b
import os
import jax
import functools
import matplotlib.pyplot as plt
import pathlib

bop_ycb_dir = os.path.join(b.utils.get_assets_dir(), "bop/ycbv")
rgbd, gt_ids, gt_poses, masks = b.utils.ycb_loader.get_test_img('52', '1', bop_ycb_dir)
fig = b.viz_depth_image(rgbd.depth)
fig.savefig("depth.png", **b.saveargs)
fig = b.viz_rgb_image(rgbd.rgb)
fig.savefig("rgb.png", **b.saveargs)

fig = plt.figure()
ax = fig.add_subplot(1,2,1)
b.add_rgb_image(ax, rgbd.rgb)
ax.set_title("RGB")
ax = fig.add_subplot(1,2,2)
b.add_depth_image(ax, rgbd.depth)
ax.set_title("DEPTH")
fig.savefig("fig.png", **b.saveargs)