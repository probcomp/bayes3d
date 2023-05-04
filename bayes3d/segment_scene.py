import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import os, sys
import random

import bayes3d

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

BACKGROUND_SUBTRACTION_INTERFACE = None
SEGMENTATION_NETWORK = None
SEGMENTATION_CROP_NETWORK = None


def segment_scene(rgb_original, point_cloud_image, intrinsics, depth_original=None, intrinsics_original=None, use_nn=False,  viz=True):
    far = intrinsics.far
    h,w = intrinsics.height, intrinsics.width


    foreground_mask = get_foreground_mask(rgb_original)[...,None]   # h x w x 1
    foreground_mask_scaled = bayes3d.utils.resize(np.array(foreground_mask), h,w)
    
    point_cloud_image_above_table = (
        point_cloud_image * 
        foreground_mask_scaled[..., None]
    )

    segmentation_image_cluster = bayes3d.utils.segment_point_cloud_image(
        point_cloud_image_above_table, threshold=0.04, min_points_in_cluster=40
    )

    if use_nn:
        assert depth_original is not None
        orig_fx, orig_fy, orig_cx, orig_cy = intrinsics_original.fx, intrinsics_original.fy, intrinsics_original.cx, intrinsics_original.cy
        segmentation_image_nn = bayes3d.segment_scene.get_segmentation_from_img(rgb_original, depth_original, foreground_mask[:,:,0], orig_fx, orig_fy, orig_cx, orig_cy)
        segmentation_image_nn = bayes3d.utils.resize(np.array(segmentation_image_nn), h,w)

        # process the final segmentation image based on clustering and nn results
        num_objects = int(segmentation_image_cluster.max()) + 1

        if segmentation_image_nn is None:
            warnings.warn("Segmentation NN failed; using clustering results")
            final_segmentation_image = segmentation_image_cluster 
        else:    
            final_segmentation_image = np.zeros(segmentation_image_cluster.shape) - 1
            max_cluster_id = 0
            for cluster_id in range(num_objects):
                print("Processing cluster id = ", cluster_id)

                cluster_region = segmentation_image_cluster == cluster_id

                cluster_region_nn_pred = segmentation_image_cluster[cluster_region]
                cluster_region_nn_pred_items = set(np.unique(cluster_region_nn_pred)) - {-1}

                if len(cluster_region_nn_pred_items) == 1:
                    # print("No further segmentation:cluster id ", max_cluster_id)
                    final_segmentation_image[cluster_region] = max_cluster_id 
                    max_cluster_id += 1
                else:  # split region 
                    nn_segmentation = segmentation_image_cluster[cluster_region]
                    final_segmentation_image[cluster_region] = nn_segmentation - nn_segmentation.min() + max_cluster_id
                    # print("Extra segmentation: cluster id ", np.unique(final_segmentation_image[cluster_region]))

                    max_cluster_id = final_segmentation_image[cluster_region].max() + 1

    else:
        final_segmentation_image = segmentation_image_cluster

    viz_image = None
    if viz:
        unique_vals = np.unique(final_segmentation_image)
        unique_vals = unique_vals[unique_vals != -1]
        images = [bayes3d.viz.get_depth_image( 1.0 * (final_segmentation_image==i), max=1.0) for i in unique_vals]
        viz_image = bayes3d.viz.multi_panel(
            [
                bayes3d.viz.resize_image(bayes3d.viz.get_rgb_image(rgb_original, 255.0),h,w),
                bayes3d.viz.resize_image(bayes3d.viz.get_rgb_image(rgb_original * foreground_mask, 255.0),h,w),
                bayes3d.viz.get_depth_image(point_cloud_image[:,:,2],  max=far),
                bayes3d.viz.get_depth_image(point_cloud_image_above_table[:,:,2],  max=far),
                bayes3d.viz.get_depth_image(final_segmentation_image + 1, max=final_segmentation_image.max() + 1),
                *images
            ],
            labels=["RGB", "RGB Masked", "Depth", "Above Table", "Segmentation : {:d}".format(int(final_segmentation_image.max() + 1)), *[f"{i}" for i in unique_vals] ] ,
        )
    return final_segmentation_image, foreground_mask, viz_image

def compute_xyz(depth_img, fx, fy, px, py, height, width):
    indices = util_.build_matrix_of_indices(height, width)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # Shape: [H x W x 3]
    return xyz_img

def reset(seed):
    print(f"resetting to seed {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_foreground_mask(rgba_array):
    """
    Run a Tracer B7 model to process away the image background.
    Return a mask array that isolates the objects.
    """
    global BACKGROUND_SUBTRACTION_INTERFACE
    # Check doc strings for more information
    if BACKGROUND_SUBTRACTION_INTERFACE is None:
        BACKGROUND_SUBTRACTION_INTERFACE = HiInterface(object_type="object",  # Can be "object" or "hairs-like".
                                batch_size_seg=5,
                                batch_size_matting=1,
                                device='cuda' if torch.cuda.is_available() else 'cpu',
                                seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
                                matting_mask_size=2048,
                                trimap_prob_threshold=220,#231,
                                trimap_dilation=15,
                                trimap_erosion_iters=20,
                                fp16=False)

    images_without_background = BACKGROUND_SUBTRACTION_INTERFACE([Image.fromarray(rgba_array)])  # TODO just take in input from sscript
    img_wo_bg = images_without_background[0]

    # create the object mask
    img_arr_wo_bg = np.array(img_wo_bg)
    assert img_arr_wo_bg.shape[-1] == 4

    mask = img_arr_wo_bg[:,:,3]  # enforce alpha threshold to create B/W mask
    mask[mask > 0] = 255
    return 1.0 * (mask > 0)


def get_segmentation_from_img(
                    rgb_array, depth_array, mask_array, fx, fy, cx, cy, 
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
    global SEGMENTATION_NETWORK
    global SEGMENTATION_CROP_NETWORK
    
    if not randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        reset(cfg.RNG_SEED)

    if SEGMENTATION_NETWORK is None:
        if cfg_file is not None:
            cfg_from_file(cfg_file)

        if len(cfg.TEST.CLASSES) == 0:
            cfg.TEST.CLASSES = cfg.TRAIN.CLASSES

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

        SEGMENTATION_NETWORK = networks.__dict__[network_name](num_classes, cfg.TRAIN.NUM_UNITS, network_data).cuda(device=cfg.device)
        SEGMENTATION_NETWORK = torch.nn.DataParallel(SEGMENTATION_NETWORK, device_ids=[cfg.gpu_id]).cuda(device=cfg.device)
        cudnn.benchmark = True
        SEGMENTATION_NETWORK.eval()

        if pretrained_crop:
            network_data_crop = torch.load(pretrained_crop)
            SEGMENTATION_CROP_NETWORK = networks.__dict__[network_name](num_classes, cfg.TRAIN.NUM_UNITS, network_data_crop).cuda(device=cfg.device)
            SEGMENTATION_CROP_NETWORK = torch.nn.DataParallel(SEGMENTATION_CROP_NETWORK, device_ids=[cfg.gpu_id]).cuda(device=cfg.device)
            SEGMENTATION_CROP_NETWORK.eval()
        else:
            SEGMENTATION_CROP_NETWORK = None


    # bgr image
    rgb = rgb_array[:, :, :3]
    im = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
 
    camera_params = {'fx':fx,
                    'fy':fy,
                    'x_offset':cx,
                    'y_offset':cy
                    }

    # depth image
    depth_img = depth_array
    depth = depth_img.astype(np.float32)

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

    out_label, out_label_refined = test_sample(sample, SEGMENTATION_NETWORK, SEGMENTATION_CROP_NETWORK, img_name="tmp")
    out_label_refined -= 1   # -1 for table, 0,1,2... for objects
    return out_label_refined[0]  # for a single image, return first element of [1, h, w] array


def get_segmentation(rgb_array, depth_array, fx, fy, cx, cy):
    rgb_array = rgb_array[:,:,:3].copy()
    mask_array = get_foreground_mask(rgb_array)
    segmentation_array = get_segmentation_from_img(rgb_array, depth_array, mask_array, fx, fy, cx, cy)

    print("retrieved segmentation array")
    return mask_array, segmentation_array
