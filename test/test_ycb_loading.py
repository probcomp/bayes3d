import jax.numpy as jnp
import bayes3d as b
import bayes3d.utils.ycb_loader
import trimesh
import os
from tqdm import tqdm


def test_ycb_loading():
    bop_ycb_dir = os.path.join(b.utils.get_assets_dir(), "bop/ycbv")
    rgbd, gt_ids, gt_poses, masks = b.utils.ycb_loader.get_test_img('52', '1', bop_ycb_dir)

    b.setup_renderer(rgbd.intrinsics, num_layers=1)

    model_dir =os.path.join(b.utils.get_assets_dir(), "bop/ycbv/models")
    for idx in range(1,22):
        b.RENDERER.add_mesh_from_file(os.path.join(model_dir,"obj_" + "{}".format(idx).rjust(6, '0') + ".ply"),scaling_factor=1.0/1000.0)

    reconstruction_depth = b.RENDERER.render(gt_poses, gt_ids)[:,:,2]
    match_fraction = (jnp.abs(rgbd.depth - reconstruction_depth) < 0.05).mean()
    assert match_fraction > 0.2
