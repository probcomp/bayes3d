import pycolmap
import jax3dp3 as j
import os

rgbd, gt_ids, gt_poses, masks = j.ycb_loader.get_test_img('52', '1', "/home/nishadgothoskar/data/bop/ycbv")
rgbd2, gt_ids, gt_poses, masks = j.ycb_loader.get_test_img('52', '643', "/home/nishadgothoskar/data/bop/ycbv")

os.makedirs("imgs",exist_ok=True)
j.get_rgb_image(rgbd.rgb).save("imgs/1.png")
j.get_rgb_image(rgbd2.rgb).save("imgs/2.png")

os.makedirs("output",exist_ok=True)
database_path = "/home/nishadgothoskar/jax3dp3/output/database.db"
output_path = "/home/nishadgothoskar/jax3dp3/output"
image_dir = "imgs"

pycolmap.extract_features(database_path, image_dir)

pycolmap.match_exhaustive(database_path)

maps = pycolmap.incremental_mapping(database_path, image_dir, output_path)

maps[0].write(output_path)

mvs_path = os.path.join(output_path, "mvs")
# dense reconstruction
pycolmap.undistort_images(mvs_path, output_path, image_dir)
pycolmap.patch_match_stereo(mvs_path)  # requires compilation with CUDA
pycolmap.stereo_fusion(mvs_path / "dense.ply", mvs_path)