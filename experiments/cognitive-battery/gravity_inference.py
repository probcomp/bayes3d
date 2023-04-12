import os
from glob import glob
from gravity_model import gravity_model

DATA_PREFIX = os.environ.get("JAX3DP3_DATA_PATH", "data/")
MESHES_PATH = os.path.join(DATA_PREFIX, "gravity_data/meshes")

condition = "1_tubes"
video_paths = glob(f"{DATA_PREFIX}/gravity_data/videos/{condition}/**")

for video_path in video_paths:
    out_path = f"out/{video_path.replace(DATA_PREFIX, '').replace('/', '_')}.gif"
    pred = gravity_model(video_path, MESHES_PATH, out_path=out_path)
