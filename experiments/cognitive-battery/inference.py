import os
import warnings
from glob import glob

import cog_utils as utils
import config
import jax.numpy as jnp
import sklearn.metrics as skm
from model import model

from typing import Tuple

warnings.filterwarnings("ignore")

DATA_PREFIX = os.environ.get("JAX3DP3_DATA_PATH", "data/")
MESHES_PATH = os.path.join(DATA_PREFIX, "{experiment}_data/meshes")

metrics = {
    "accuracy": skm.accuracy_score,
    "precision": lambda y_true, y_pred: skm.precision_score(
        y_true, y_pred, average="macro"
    ),
    "recall": lambda y_true, y_pred: skm.recall_score(y_true, y_pred, average="macro"),
    "f1": lambda y_true, y_pred: skm.f1_score(y_true, y_pred, average="macro"),
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
    video_paths = glob(f"{DATA_PREFIX}/{experiment}_data/videos/{condition}/**")[:10]
    video_paths = [f"{DATA_PREFIX}/{experiment}_data/videos/{condition}/36"]

    n = len(video_paths)
    preds, labels = jnp.empty(n), jnp.empty(n)
    for i, video_path in enumerate(video_paths):
        pred, label = run_inference(experiment, video_path)
        preds[i] = pred
        labels[i] = label

    for m in metrics:
        metric_value = metrics[m](labels, preds)
        print(m, metric_value)
