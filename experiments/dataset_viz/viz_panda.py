
import numpy as np
import cv2
import jax3dp3.transforms_3d as t3d
import os
import jax
import jax.numpy as jnp
import trimesh
import time
import jax3dp3
import pickle
import os

panda_dataset_path = os.path.join(jax3dp3.utils.get_assets_dir(), "panda_dataset")
file = open(os.path.join(panda_dataset_path, "scene_1.pkl"),'rb')
all_data = pickle.load(file)
file.close()

frames = []

for data in all_data:
    rgb = data["rgb"]
    rgb_viz = jax3dp3.viz.get_rgb_image(rgb, 255.0)
    frames.append(rgb_viz)

jax3dp3.viz.make_gif_from_pil_images(frames, "panda_2.gif")

from IPython import embed; embed()