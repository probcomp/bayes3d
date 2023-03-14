import sys

densefusion_path = "../jax3dp3/posecnn-pytorch/PoseCNN-PyTorch"
sys.path.append(densefusion_path)   # TODO cleaner import / add to path
import os
import easydict
import pickle
import torch 
import numpy as np
import random
import tools._init_paths
from tools.test_images_utils import env_setup_posecnn, get_image_posecnn, run_posecnn, get_image_densefusion, env_setup_densefusion, run_DenseFusion
from fcn.config import cfg, cfg_from_file



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


posecnn_args, posecnn_cfg, densefusion_args = setup_baseline()

dataset_cfg, posecnn_network = env_setup_posecnn(posecnn_args, posecnn_cfg)
df_estimator, df_refiner, class_names, cld = env_setup_densefusion(densefusion_args)

# Input here
test_filename_pik = '/home/ubuntu/jax3dp3/jax3dp3/posecnn-pytorch/PoseCNN-PyTorch/datasets/pandas/data/new_panda/demo2_nolight-0.pkl'
test_filename_pik = '/home/ubuntu/jax3dp3/jax3dp3/posecnn-pytorch/PoseCNN-PyTorch/datasets/pandas/data/new_panda/strawberry_error-0.pkl'
# test_filename_pik = '/home/ubuntu/jax3dp3/jax3dp3/posecnn-pytorch/PoseCNN-PyTorch/datasets/pandas/data/new_panda/knife_sim-0.pkl'

scene_name = test_filename_pik.split('/')[-1].split('.')[0]
print(scene_name)


# Load Test Image
with open(test_filename_pik, 'rb') as file:
    test_data = pickle.load(file)
image_color_bgr, image_depth, meta_data = get_image_posecnn(test_data)  # BGR
image_color_bgr = image_color_bgr[:,:,:3]
image_color_rgb, _, _ = get_image_densefusion(test_data)
image_color_rgb = image_color_rgb[:,:,:3]


###########
# Run PoseCNN + DenseFusion
###########
print(f"\n Running models on {test_filename_pik}..")
posecnn_meta = run_posecnn(image_color_bgr, image_depth, meta_data, posecnn_network, dataset_cfg, posecnn_cfg)    

# prediction_results is the final pose estimation result after refinement (class_name: {'class_id':int, 'rot_q': quaternion, 'tr': translation})
prediction_results =  run_DenseFusion(image_color_rgb, image_depth, meta_data,
                                        df_estimator, df_refiner, 
                                        class_names=class_names, 
                                        cld=cld,
                                        densefusion_args=densefusion_args,
                                        scene_frame_name=scene_name, 
                                        posecnn_meta=posecnn_meta)


# intrinsics = j.Intrinsics(
#     height=300,
#     width=300,
#     fx=200.0, fy=200.0,
#     cx=150.0, cy=150.0,
#     near=0.001, far=50.0
# )

# input: rgb image, depth image, intrinsics (Intrinsics) (assume factor_depth is always 1)
# output: [(id, pos, quat)]

from IPython import embed; embed()
