import jax.numpy as jnp
import jax3dp3 as j
import trimesh
import os
import numpy as np
import pybullet_planning


# rgbd, gt_ids, gt_poses, masks = j.ycb_loader.get_test_img('52', '1', "/home/nishadgothoskar/data/bop/ycbv")

# model_dir = "/home/nishadgothoskar/data/bop/ycbv/models"
# mesh_paths = []
# for idx in range(1,22):
#     mesh_paths.append(os.path.join(model_dir,"obj_" + "{}".format(idx).rjust(6, '0') + ".ply") )

# paths = [mesh_paths[i] for i in gt_ids]

intrinsics = j.Intrinsics(
    height=300,
    width=300,
    fx=2000.0, fy=2000.0,
    cx=150.0, cy=150.0,
    near=0.001, far=50.0
)


# top_level_dir = os.path.dirname(os.path.dirname(pybullet_planning.__file__))
# top_level_dir = os.path.dirname(os.path.dirname(j.__file__))
# knife_dir = "assets/sample_objs/ycb_knife/textured.obj"

# mesh_names = ["knife", "spoon", "cracker_box", "strawberry", "mustard_bottle", "sugar_box","banana"]
# model_paths = [
#     os.path.join(top_level_dir, knife_dir),
# ]

top_level_dir = os.path.dirname(os.path.dirname(pybullet_planning.__file__))
mesh_names = ["knife", "spoon", "cracker_box", "strawberry", "mustard_bottle", "sugar_box","banana"]
model_paths = [
    os.path.join(top_level_dir,"models/srl/ycb/032_knife/textured.obj"),
]

gt_poses = jnp.array([
    j.t3d.transform_from_pos(jnp.array([0.0, 0.0, 2.0]))
])

rgb, segmentation, depth = j.kubric_interface.render_kubric(model_paths, gt_poses, jnp.eye(4), intrinsics, scaling_factor=1.0)


rgb_viz = j.get_rgb_image(rgb)
depth_viz = j.get_depth_image(depth, max=10.0)
seg_viz = j.get_depth_image(segmentation, max=segmentation.max())

j.multi_panel(
    [
        rgb_viz,
        depth_viz,
        seg_viz
    ]
).save("test_kubric.png")

renderer = j.Renderer(intrinsics)
renderer.add_mesh_from_file(model_paths[0])
img = renderer.render_multiobject(gt_poses, [0])
depth_viz = j.get_depth_image(img[:,:,2], max=intrinsics.far).save("mine.png")


# j.setup_visualizer()
# j.show_cloud("1",j.t3d.unproject_depth(depth[:,:,0], intrinsics).reshape(-1,3))
from IPython import embed; embed()
