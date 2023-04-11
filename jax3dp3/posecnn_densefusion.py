import sys
import os
densefusion_path = f"{os.path.dirname(os.path.abspath(__file__))}/posecnn-pytorch/PoseCNN-PyTorch"  
sys.path.append(densefusion_path)   # TODO cleaner import / add to path
import easydict
import pickle
import torch 
import numpy as np
import random
import tools._init_paths
from tools.test_images_utils import env_setup_posecnn, get_image_posecnn, run_posecnn, get_image_densefusion, env_setup_densefusion, run_DenseFusion
from fcn.config import cfg, cfg_from_file

import jax3dp3 as j

def state_reset(seed=100):
    print(f"resetting to seed {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True 


def setup_baseline():
    #############
    # Setup PoseCNN
    #############
    state_reset()
    posecnn_args = easydict.EasyDict({
        'gpu_id':0, 
        'pretrained':f'{densefusion_path}/data/trained_checkpoints/posecnn/ycb_object/vgg16_ycb_object_self_supervision_epoch_8.checkpoint.pth', 
        'cfg_file':f'{densefusion_path}/experiments/cfgs/ycb_object.yml', 
        'dataset_name':'ycb_object_test', 
        # 'depth_name':None, # will process results from pik instead 
        'datadir':f'{densefusion_path}/datasets/pandas/data/new_panda/', # provide a dir containing piks
        'resdir':f'{densefusion_path}/datasets/pandas/data/results_PoseCNN_pandas/', 
        'network_name':'posecnn', 
        'background_name':None,
        'meta_file':None, 
        'pretrained_encoder':None, 
        'codebook':None, 
    })
    if posecnn_args.cfg_file is not None:
        cfg_from_file(posecnn_args.cfg_file)  # creates variable `cfg`
        posecnn_cfg = cfg

    #############
    # Setup DenseFusion
    #############
    this_dir = os.getcwd()
    densefusion_args = easydict.EasyDict({
        # 'dataset_root': f'{densefusion_path}/datasets/pandas',
        'model': f'{densefusion_path}/data/trained_checkpoints/densefusion/ycb/pose_model_26_0.012863246640872631.pth',
        'refine_model': f'{densefusion_path}/data/trained_checkpoints/densefusion/ycb/pose_refine_model_69_0.009449292959118935.pth',
        'dataset_config_dir':f'{densefusion_path}/datasets/pandas/dataset_config',
        # 'ycb_toolbox_dir':None,  
        'result_wo_refine_dir':f'{this_dir}/Densefusion_wo_refine_result',  
        'result_refine_dir':f'{this_dir}/Densefusion_iterative_result' 
    })


    return posecnn_args, posecnn_cfg, densefusion_args


class DenseFusion(object):
    def __init__(self,):
        self.posecnn_args, self.posecnn_cfg, self.densefusion_args = setup_baseline()
        self.dataset_cfg, self.POSECNN_NETWORK = env_setup_posecnn(self.posecnn_args, self.posecnn_cfg)
        self.DF_ESTIMATOR, self.DF_REFINER, self.class_names, self.cld = env_setup_densefusion(self.densefusion_args)


    def get_densefusion_results(self, rgb_image, depth_image, intrinsics, scene_name="out", factor_depth=1):
        rgb_image = rgb_image[:,:,:3]
        K = j.camera.K_from_intrinsics(intrinsics)
        bgr_image = np.copy(rgb_image)
        bgr_image[:,:,2] = bgr_image[:,:,0]
        bgr_image[:,:,0] = rgb_image[:,:,2]

        meta_data = dict({'factor_depth': factor_depth, 'intrinsic_matrix': K})

        print(f"\n Running models on {scene_name}..")
        posecnn_meta = run_posecnn(bgr_image, depth_image, meta_data, self.POSECNN_NETWORK, self.dataset_cfg, self.posecnn_cfg)    
        print(f"\n Running dense fusion models on {scene_name}..")
        prediction_results =  run_DenseFusion(rgb_image, depth_image, meta_data,
                                            self.DF_ESTIMATOR, self.DF_REFINER, 
                                            class_names=self.class_names, 
                                            cld=self.cld,
                                            densefusion_args=self.densefusion_args,
                                            scene_frame_name=scene_name, 
                                            posecnn_meta=posecnn_meta)    

        return prediction_results

 