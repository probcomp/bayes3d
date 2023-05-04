import jax.numpy as jnp
import bayes3d as j
import trimesh
import os
import numpy as np

rgbd, gt_ids, gt_poses, masks = j.ycb_loader.get_test_img('52', '1', "/home/nishadgothoskar/data/bop/ycbv")

model_dir = "/home/nishadgothoskar/models"
mesh_paths = []
model_names = j.ycb_loader.MODEL_NAMES
for name in model_names:
    mesh_paths.append(
        os.path.join(model_dir,name,"textured.obj")
    )

paths = [mesh_paths[i] for i in gt_ids]
meshes = []
for p in paths:
    meshes.append(trimesh.load(p))
rgb, depth = j.blenderproc.render_blenderproc(paths, gt_poses, jnp.eye(4), rgbd.intrinsics, scaling_factor=1.0)

j.get_rgb_image(rgb).save("rgb.png")




from IPython import embed; embed()



from IPython import embed; embed()
