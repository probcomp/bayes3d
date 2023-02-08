import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

import pickle

import os, sys
import glob

from PIL import Image
import numpy as np
import cv2


background_removal_dir = os.path.join(os.path.dirname(__file__), "segmentation/image_background_remove_tool")
segmentation_dir = os.path.join(os.path.dirname(__file__), "segmentation/object_segmentation/UnseenObjectClustering")  # TODO name cleanup


sys.path.append(background_removal_dir)  # TODO cleanup dirs to avoid possible conflicts
from carvekit.api.high import HiInterface

sys.path.append(segmentation_dir)  # TODO cleanup dirs to avoid possible conflicts
import tools._init_paths
from fcn.test_dataset import test_sample
from fcn.config import cfg, cfg_from_file, get_output_dir
import networks
from utils.blob import pad_im
from utils import mask as util_




def compute_xyz(depth_img, fx, fy, px, py, height, width):
    indices = util_.build_matrix_of_indices(height, width)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # Shape: [H x W x 3]
    return xyz_img


def prepare_segmentation_data(rgb_array, depth_array, mask_array, K, factor_depth=1):
    # bgr image
    rgb = rgb_array
    im = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    fx, fy, cx, cy = K[0][0],K[1][1],K[0][2],K[1][2]
    camera_params = {'fx':fx,
                    'fy':fy,
                    'x_offset':cx,
                    'y_offset':cy
                    }

    # depth image
    depth_img = depth_array
    depth = depth_img.astype(np.float32) / factor_depth

    height = depth.shape[0]
    width = depth.shape[1]
    fx = camera_params['fx']
    fy = camera_params['fy']
    px = camera_params['x_offset']
    py = camera_params['y_offset']

    # process masking
    rgb[mask_array == 0] = np.array([0,0,0])
    im = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    xyz_img = compute_xyz(depth, fx, fy, px, py, height, width)

    im_tensor = torch.from_numpy(im) / 255.0
    pixel_mean = torch.tensor(cfg.PIXEL_MEANS / 255.0).float()
    im_tensor -= pixel_mean
    image_blob = im_tensor.permute(2, 0, 1)
    sample = {'image_color': image_blob.unsqueeze(0)}

    depth_blob = torch.from_numpy(xyz_img).permute(2, 0, 1)
    sample['depth'] = depth_blob.unsqueeze(0)

    return sample


def get_scene_wo_bg(rgba_array, viz=False, save_filename=None):
    """
    Run a Tracer B7 model to process away the image background.
    Return a mask array that isolates the objects.
    """

    # Check doc strings for more information
    interface = HiInterface(object_type="object",  # Can be "object" or "hairs-like".
                            batch_size_seg=5,
                            batch_size_matting=1,
                            device='cuda' if torch.cuda.is_available() else 'cpu',
                            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
                            matting_mask_size=2048,
                            trimap_prob_threshold=220,#231,
                            trimap_dilation=15,
                            trimap_erosion_iters=20,
                            fp16=False)


    images_without_background = interface([Image.fromarray(rgba_array)])  # TODO just take in input from sscript
    img_wo_bg = images_without_background[0]

    # create the object mask
    img_arr_wo_bg = np.array(img_wo_bg)
    assert img_arr_wo_bg.shape[-1] == 4

    mask = img_arr_wo_bg[:,:,3]  # enforce alpha threshold to create B/W mask
    mask[mask > 0] = 255

    if viz:
        img_wo_bg.save(f'{save_filename}_no_background.png')
        mask_img = Image.fromarray(mask)
        mask_img.save(f'{save_filename}_mask.png')

    return mask


def get_segmentation_from_img_wo_bg(segmentation_data, 
                    test_name = 'TEST_NAME',
                    gpu_id = 0,
                    network_name = 'seg_resnet34_8s_embedding',
                    pretrained = f'{segmentation_dir}/data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth',
                    pretrained_crop = f'{segmentation_dir}/data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth',
                    cfg_file = f'{segmentation_dir}/experiments/cfgs/seg_resnet34_8s_embedding_cosine_rgbd_add_tabletop_pandas.yml',
                    randomize = False,
                    ):
    """
    Run the UOIS model to segment the scene, 
    placing the background removal mask on the rgb image  
    """
    
    if cfg_file is not None:
        cfg_from_file(cfg_file)

    if len(cfg.TEST.CLASSES) == 0:
        cfg.TEST.CLASSES = cfg.TRAIN.CLASSES

    if not randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)

    # device
    cfg.gpu_id = 0
    cfg.device = torch.device('cuda:{:d}'.format(cfg.gpu_id))
    cfg.instance_id = 0
    num_classes = 2
    cfg.MODE = 'TEST'
    print('GPU device {:d}'.format(gpu_id))


    # prepare network
    if pretrained:
        network_data = torch.load(pretrained)
        print("=> using pre-trained network '{}'".format(pretrained))
    else:
        network_data = None
        print("no pretrained network specified")
        sys.exit()

    network = networks.__dict__[network_name](num_classes, cfg.TRAIN.NUM_UNITS, network_data).cuda(device=cfg.device)
    network = torch.nn.DataParallel(network, device_ids=[cfg.gpu_id]).cuda(device=cfg.device)
    cudnn.benchmark = True
    network.eval()

    if pretrained_crop:
        network_data_crop = torch.load(pretrained_crop)
        network_crop = networks.__dict__[network_name](num_classes, cfg.TRAIN.NUM_UNITS, network_data_crop).cuda(device=cfg.device)
        network_crop = torch.nn.DataParallel(network_crop, device_ids=[cfg.gpu_id]).cuda(device=cfg.device)
        network_crop.eval()
    else:
        network_crop = None

    
    out_label, out_label_refined = test_sample(segmentation_data, network, network_crop, img_name=test_name)
    

    return out_label_refined


def get_segmentation(rgb_array, depth_array, intrinsics, test_name, factor_depth=1):
    if not rgb_array.shape[-1] == 3:
        rgb_array = rgb_array[:,:,:3]
    mask_array = get_scene_wo_bg(rgb_array, save_filename=test_name)
    test_data = prepare_segmentation_data(rgb_array, depth_array, mask_array, intrinsics, factor_depth=factor_depth)
    segmentation_array = get_segmentation_from_img_wo_bg(test_data, test_name)

    final_segmentation_array = segmentation_array - 1   # -1 for table, 0,1,2... for objects
    print("retrieved segmentation array")
    return final_segmentation_array

