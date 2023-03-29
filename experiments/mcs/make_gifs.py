import jax3dp3 as j
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
import os
import jax
import time
from tqdm import tqdm
import jax
import matplotlib.pyplot as plt

import glob

scene_regex = os.path.join(j.utils.get_assets_dir(), "mcs_scene_jsons", "eval_6_validation", "passive_physics_*.json")
files = glob.glob(scene_regex)
for scene_path in files:
    scene_name = scene_path.split("/")[-1]
    images = j.physics.load_mcs_scene_data(scene_path)
    j.make_gif(
        [j.multi_panel([j.get_rgb_image(image.rgb)], [f"{i} / {len(images)}"]) for (i, image) in enumerate(images)],
        f"movies/{scene_name}.gif"
    )
