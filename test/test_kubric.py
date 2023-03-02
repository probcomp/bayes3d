import jax.numpy as jnp
import jax3dp3 as j
import trimesh
import os
import numpy as np
import pybullet_planning


rgbd, gt_ids, gt_poses, masks = j.ycb_loader.get_test_img('52', '1', "/home/nishadgothoskar/data/bop/ycbv")

model_dir = "/home/nishadgothoskar/models"
mesh_paths = []
model_names = j.ycb_loader.MODEL_NAMES
for name in model_names:
    mesh_paths.append(
        os.path.join(model_dir,name,"textured.obj")
    )

paths = []
for i in gt_ids:
    paths.append(mesh_paths[i])

intrinsics = j.Intrinsics(
    rgbd.intrinsics.height, rgbd.intrinsics.width,
    rgbd.intrinsics.fx, rgbd.intrinsics.fx,
    rgbd.intrinsics.width/2, rgbd.intrinsics.height/2,
    rgbd.intrinsics.near, rgbd.intrinsics.far
)
rgb, seg, depth = j.kubric_interface.render_kubric(paths, gt_poses, jnp.eye(4), intrinsics, scaling_factor=1.0)

rgb_viz = j.get_rgb_image(rgb)
depth_viz = j.get_depth_image(depth, max=intrinsics.far)
seg_viz = j.get_depth_image(seg, max=seg.max())
j.multi_panel(
    [
        rgb_viz,
        depth_viz,
        seg_viz
    ]
).save("test_kubric.png")

renderer = j.Renderer(intrinsics)
for p in paths:
    renderer.add_mesh_from_file(p)
img = renderer.render_multiobject(gt_poses, jnp.arange(len(paths)))
depth_viz = j.get_depth_image(img[:,:,2], max=intrinsics.far).save("mine.png")


j.setup_visualizer()
j.show_cloud("1", j.t3d.unproject_depth(depth, intrinsics).reshape(-1,3),color=j.RED)
j.show_cloud("2", j.t3d.unproject_depth(img[:,:,2], intrinsics).reshape(-1,3), color=j.BLUE)

from IPython import embed; embed()
