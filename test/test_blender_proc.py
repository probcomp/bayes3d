import jax.numpy as jnp
import jax3dp3 as j
import trimesh
import os
import numpy as np

rgb, depth, cam_pose, intrinsics, gt_ids, gt_poses, masks = j.ycb_loader.get_test_img('52', '1', "/home/nishadgothoskar/data/bop/ycbv")

model_dir = "/home/nishadgothoskar/data/bop/ycbv/models"
mesh_paths = []
for idx in range(1,22):
    mesh_paths.append(os.path.join(model_dir,"obj_" + "{}".format(idx).rjust(6, '0') + ".ply") )

paths = [mesh_paths[i] for i in gt_ids]

rgb, depth = j.blenderproc.render_blenderproc(paths, gt_poses, jnp.eye(4), intrinsics, scaling_factor=1.0/1000.0)
j.get_rgb_image(rgb).save("rgb.png")




from IPython import embed; embed()



from IPython import embed; embed()
