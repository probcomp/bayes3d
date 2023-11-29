import bayes3d as b
import bayes3d.colmap
import glob
from pathlib import Path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("movie_path", 
                        help="Path to movie file", 
                        type=str)
args = parser.parse_args()

b.setup_visualizer()

movie_file_path = Path(args.movie_path)

dataset_path = Path(b.utils.get_assets_dir()) / Path(movie_file_path.name + "_colmap_dataset")
input_path = dataset_path / Path("input")
input_path.mkdir(parents=True, exist_ok=True)
b.utils.video_to_images(movie_file_path, input_path)

import subprocess

assets_dir = Path(b.utils.get_assets_dir())
script_path = assets_dir.parent / Path("scripts/run_colmap.py")
import subprocess
subprocess.run([f"python {str(script_path)} -s {str(dataset_path)}"],shell=True) 


image_paths = sorted(glob.glob(str(input_path / Path("*.jpg"))))
print(len(image_paths))
images = [b.viz.load_image_from_file(f) for f in image_paths]
# b.make_gif_from_pil_images(images, "input.gif")
(positions, colors, normals), train_cam_infos = b.colmap.readColmapSceneInfo(
    dataset_path,
    "images",
    False
)

train_cam_infos[0].FovY

b.clear()
scaling_factor = 0.1
poses = [
    b.transform_from_rot_and_pos(i.R, i.T * scaling_factor) for i in train_cam_infos
]

b.show_cloud("cloud", positions * scaling_factor)
for (i,p) in enumerate(poses):
    b.show_pose(f"{i}", p)
