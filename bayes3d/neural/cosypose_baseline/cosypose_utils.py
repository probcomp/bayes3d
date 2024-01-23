import os
import signal
import subprocess
import sys

import numpy as np

cosypose_path = (
    f"{os.path.dirname(os.path.abspath(__file__))}/cosypose_baseline/cosypose"
)
sys.path.append(cosypose_path)  # TODO cleaner import / add to path

import time

import torch
import yaml

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.synchronize()

COSYPOSE_CONDA_ENV_NAME = "cosypose"
COSYPOSE_MODEL = None


class CosyPose(object):
    def __init__(
        self,
        detector_run_id="detector-bop-ycbv-synt+real--292971",
        coarse_run_id="coarse-bop-ycbv-synt+real--822463",
        refiner_run_id="refiner-bop-ycbv-synt+real--631598",
    ) -> None:
        self.detector, self.pose_predictor = self.get_models(
            detector_run_id, coarse_run_id, refiner_run_id
        )

    def load_detector(self, run_id):
        print("EXPDIR=", EXP_DIR)
        run_dir = EXP_DIR / run_id
        cfg = yaml.load((run_dir / "config.yaml").read_text(), Loader=yaml.FullLoader)
        cfg = check_update_config_detector(cfg)
        label_to_category_id = cfg.label_to_category_id
        model = create_model_detector(cfg, len(label_to_category_id))
        ckpt = torch.load(run_dir / "checkpoint.pth.tar")
        ckpt = ckpt["state_dict"]
        model.load_state_dict(ckpt)
        model = model.cuda().eval()
        model.cfg = cfg
        model.config = cfg
        model = Detector(model)
        return model

    def load_pose_models(self, coarse_run_id, refiner_run_id=None, n_workers=8):
        run_dir = EXP_DIR / coarse_run_id
        cfg = yaml.load((run_dir / "config.yaml").read_text(), Loader=yaml.FullLoader)
        cfg = check_update_config_pose(cfg)
        # object_ds = BOPObjectDataset(BOP_DS_DIR / 'tless/models_cad')
        object_ds = make_object_dataset(cfg.object_ds_name)
        mesh_db = MeshDataBase.from_object_ds(object_ds)
        renderer = BulletBatchRenderer(object_set=cfg.urdf_ds_name, n_workers=n_workers)
        mesh_db_batched = mesh_db.batched().cuda()

        def load_model(run_id):
            if run_id is None:
                return
            run_dir = EXP_DIR / run_id
            cfg = yaml.load(
                (run_dir / "config.yaml").read_text(), Loader=yaml.FullLoader
            )
            cfg = check_update_config_pose(cfg)
            if cfg.train_refiner:
                model = create_model_refiner(
                    cfg, renderer=renderer, mesh_db=mesh_db_batched
                )
            else:
                model = create_model_coarse(
                    cfg, renderer=renderer, mesh_db=mesh_db_batched
                )
            ckpt = torch.load(run_dir / "checkpoint.pth.tar")
            ckpt = ckpt["state_dict"]
            model.load_state_dict(ckpt)
            model = model.cuda().eval()
            model.cfg = cfg
            model.config = cfg
            return model

        coarse_model = load_model(coarse_run_id)
        refiner_model = load_model(refiner_run_id)
        model = CoarseRefinePosePredictor(
            coarse_model=coarse_model, refiner_model=refiner_model
        )
        return model, mesh_db

    def get_models(self, detector_run_id, coarse_run_id, refiner_run_id):
        # load models
        detector = self.load_detector(detector_run_id)
        pose_predictor, mesh_db = self.load_pose_models(
            coarse_run_id=coarse_run_id, refiner_run_id=refiner_run_id, n_workers=4
        )
        return detector, pose_predictor

    def inference(self, image, camera_k):
        # [1,540,720,3]->[1,3,540,720]
        images = torch.from_numpy(image).cuda().float().unsqueeze_(0)
        images = images.permute(0, 3, 1, 2) / 255
        # [1,3,3]
        cameras_k = torch.from_numpy(camera_k).cuda().float().unsqueeze_(0)
        # 2D detector
        # print("start detect object.")
        box_detections = self.detector.get_detections(
            images=images,
            one_instance_per_class=False,
            detection_th=0.8,
            output_masks=False,
            mask_th=0.9,
        )
        # pose esitimition
        if len(box_detections) == 0:
            return None
        # print("start estimate pose.")
        final_preds, all_preds = self.pose_predictor.get_predictions(
            images,
            cameras_k,
            detections=box_detections,
            n_coarse_iterations=1,
            n_refiner_iterations=4,
        )

        # result: this_batch_detections, final_preds
        return final_preds


def cosypose_interface(rgb_imgs, camera_k):
    rgb_imgs = rgb_imgs[..., :3]
    if os.path.exists("/tmp/cosypose_output.npz"):
        os.remove("/tmp/cosypose_output.npz")
    else:
        print("The file does not exist")

    if len(rgb_imgs.shape) == 3:
        # unsqueeze into (1, H, W, 3) if single-image (H,W,3)
        rgb_imgs = rgb_imgs[None, :]

    np.savez("/tmp/cosypose_input.npz", rgbs=rgb_imgs, K=camera_k)

    print("Entering COSYPOSE")

    py = os.popen(f"conda run -n {COSYPOSE_CONDA_ENV_NAME} which python").read().strip()
    print(py)
    cmd = f"{py} {os.path.abspath(__file__)}"
    pro = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, shell=True, preexec_fn=os.setsid
    )

    while not os.path.exists("/tmp/cosypose_output.npz"):
        # print("waiting...")
        time.sleep(1.0)
    os.killpg(
        os.getpgid(pro.pid), signal.SIGTERM
    )  # Send the signal to all the process groups

    print("Finished COSYPOSE")
    data = np.load("/tmp/cosypose_output.npz")

    return data


if __name__ == "__main__":
    print("SUBPROCESS:", sys.version)  # expect Python 3.7.6

    # do imports here to bypass during imports from jax3dp3 __init__
    from cosypose.config import EXP_DIR
    from cosypose.datasets.datasets_cfg import make_object_dataset
    from cosypose.integrated.detector import Detector
    from cosypose.integrated.pose_predictor import CoarseRefinePosePredictor

    # Pose estimator
    from cosypose.lib3d.rigid_mesh_database import MeshDataBase
    from cosypose.rendering.bullet_batch_renderer import BulletBatchRenderer
    from cosypose.training.detector_models_cfg import (
        check_update_config as check_update_config_detector,
    )

    # Detection
    from cosypose.training.detector_models_cfg import create_model_detector
    from cosypose.training.pose_models_cfg import (
        check_update_config as check_update_config_pose,
    )
    from cosypose.training.pose_models_cfg import (
        create_model_coarse,
        create_model_refiner,
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # load model
    print("instantiated")
    COSYPOSE_MODEL = CosyPose()

    # load data
    data = np.load("/tmp/cosypose_input.npz")
    rgb_imgs, camera_k = data["rgbs"], data["K"]
    num_imgs = len(rgb_imgs)

    all_poses = []
    all_ids = []
    all_scores = []
    for i, rgb_img in enumerate(rgb_imgs):
        pred = COSYPOSE_MODEL.inference(rgb_img, camera_k)
        print(f"{i+1}/{num_imgs} inference done")

        pred_poses = np.asarray(pred.poses.cpu())
        pred_ids = [
            int(l[-3:]) - 1 for l in pred.infos.label
        ]  # ex) 'obj_000014' for GT_IDX 13
        pred_scores = [pred.infos.iloc[i].score for i in range(len(pred.infos))]

        all_poses.append(pred_poses)
        all_ids.append(pred_ids)
        all_scores.append(pred_scores)
        print(pred_poses, pred_ids, pred_scores)

    np.savez(
        "/tmp/cosypose_output.npz",
        pred_poses=np.asarray(all_poses),
        pred_ids=np.asarray(all_ids),
        pred_scores=np.asarray(all_scores),
    )
    print("saved results. exiting")
    sys.exit()
