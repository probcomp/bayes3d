import os
import warnings
from glob import glob
from typing import Tuple

import cog_utils as utils
import config
import sklearn.metrics as skm
from model import model

warnings.filterwarnings("ignore")

DATA_PREFIX = os.environ.get("JAX3DP3_DATA_PATH", "data/")
MESHES_PATH = os.path.join(DATA_PREFIX, "{experiment}_data/meshes")

metrics = {
    "accuracy": skm.accuracy_score,
    "precision": lambda yt, yp: skm.precision_score(yt, yp, average="macro"),
    "recall": lambda yt, yp: skm.recall_score(yt, yp, average="macro"),
    "f1": lambda yt, yp: skm.f1_score(yt, yp, average="macro"),
    "mae": skm.mean_absolute_error,
}


def run_inference(experiment: str, video_path: str) -> Tuple[int, int]:
    out_path = f"out/{video_path.replace(DATA_PREFIX, '').replace('/', '_')}.gif"
    pred = model(
        config=config.CONFIG_MAP[experiment],
        video_path=video_path,
        meshes_path=MESHES_PATH.format(experiment=experiment),
        out_path=out_path,
    )
    label = utils.read_label(video_path)

    return pred, label


if __name__ == "__main__":
    experiment = "gravity"
    condition = "train/1_tubes"
    video_paths = glob(f"{DATA_PREFIX}/{experiment}_data/videos/{condition}/**")

    n = len(video_paths)
    preds, labels = [], []
    for i, video_path in enumerate(video_paths):
        pred, label = run_inference(experiment, video_path)
        preds.append(pred)
        labels.append(label)

    for m in metrics:
        metric_value = metrics[m](labels, preds)
        print(m, metric_value)
