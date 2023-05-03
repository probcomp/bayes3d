import numpy as np
import os
import jax.numpy as jnp
import time
import jax3dp3.transforms_3d as t3d
import jax3dp3 as j

from c2f_scripts import *

# TODO: move these to utils
def load_dataset(dataset_idx):
    dataset_filename = f"datasets/dataset_{dataset_idx}.npz"  # npz file
    dataset_file = os.path.join(j.utils.get_assets_dir(), f"datasets/{dataset_filename}")
    data = np.load(dataset_file, allow_pickle=True)
    num_scenes_in_dataset = len(data['rgbds'])

    return data, num_scenes_in_dataset

def load_scene(dataset:np.lib.npyio.NpzFile, scene_idx):
 
    rgbd_img = dataset['rgbds'][scene_idx]
    gt_pose = dataset['poses'][scene_idx]
    gt_name = str(dataset['name'])
    gt_idx = int(dataset['id'])

    # sanity check
    assert len(np.unique(rgbd_img.segmentation[rgbd_img.segmentation != 0])) == 1, "Single object dataset should contain one unique nonzero segmentation ID"
    if gt_idx not in rgbd_img.segmentation:
        rgbd_img.segmentation[rgbd_img.segmentation != 0] = gt_idx 
    return (rgbd_img, gt_idx, jnp.asarray(gt_pose))

def save_dataset_results(dataset_idx, pred_weights, pred_poses):
    print("SAVING RESULTS...")
    gt_data, _ = load_dataset(dataset_idx)

    gt_rgbds = gt_data['rgbds'] 
    gt_poses = gt_data['poses'] 
    gt_name = str(gt_data['name'])
    gt_idx = int(gt_data['id'])
    
    np.savez(f"{RESULTS_DATA_DIR}/results_dataset_{dataset_idx}.npz", 
        gt_rgbds = gt_rgbds, 
        gt_poses = gt_poses,
        gt_name = gt_name,
        gt_idx = gt_idx,
        grid_widths = np.asarray(grid_widths),
        rot_angle_widths = np.asarray(rot_angle_widths),
        sphere_angle_widths = np.asarray(sphere_angle_widths),
        grid_params = np.asarray(grid_params),
        pred_weights = np.asarray(pred_weights),
        pred_poses = np.asarray(pred_poses)
    )


def setup_renderer(num_layers=25, scale_factor=0.3):
    print(f"Renderer with {num_layers} layers and scale factor {scale_factor}")
    sample_rgbd = load_dataset(0)[0]['rgbds'][0]

    ## setup intrinsics and renderer
    intrinsics = j.camera.scale_camera_parameters(sample_rgbd.intrinsics, scale_factor)
    renderer = j.Renderer(intrinsics, num_layers=num_layers)

    ## load models
    # model_dir = os.path.join(j.utils.get_assets_dir(), "ycb_video_models/models")
    # model_names = j.ycb_loader.MODEL_NAMES
    model_dir = os.path.join(j.utils.get_assets_dir(), "bop/ycbv/models")
    model_names = ["obj_" + f"{str(idx+1).rjust(6, '0')}.ply" for idx in range(21)]
    mesh_paths = []
    for name in model_names:
        mesh_path = os.path.join(model_dir,name)
        mesh_paths.append(mesh_path)
        model_scaling_factor = 1.0/1000.0
        renderer.add_mesh_from_file(
            mesh_path,
            scaling_factor=model_scaling_factor
        )

    print("finished renderer setup")
    return renderer


def test_scene(dataset_idx, test_idx, particles=1):
    print(f"#### Starting scene {test_idx} in Dataset {dataset_idx} ####")
    ## load data, visualize gt
    data, _ = load_dataset(dataset_idx)
    image, gt_idx, gt_pose = load_scene(data, test_idx)
    print(gt_pose)
    j.viz.get_depth_image(image.depth).save(f"{RESULTS_IMGS_DIR}/dataset_{dataset_idx}_scene_{test_idx}_gt.png")
    rendered = renderer.render_single_object(gt_pose, gt_idx)  
    viz = j.viz.get_depth_image(rendered[:,:,2], min=jnp.min(rendered), max=jnp.max(rendered[:,:,2])+0.1)
    viz.save(f"_gt_depth.png")
    j.viz.resize_image(viz, image.intrinsics.height, image.intrinsics.width).save(f"{RESULTS_IMGS_DIR}/dataset_{dataset_idx}_scene_{test_idx}_gt_depth.png")

    ## run c2f
    c2f_results = run_c2f(renderer, image, scheds, infer_id=False, infer_contact=False, particles=particles)
    c2f_results = c2f_results[0]  # single-segment, single-object
    
    print("gt:", gt_pose)
    c2f_iter = len(c2f_results)
    c2f_result = c2f_results[c2f_iter-1]  # TODO cleanup dimensions
    _, gt_idx, best_pose = c2f_result[0]  # results are sorted highest-to-lowest weight
    print(best_pose)

    rendered = renderer.render_single_object(best_pose, gt_idx)  
    viz = j.viz.get_depth_image(rendered[:,:,2], min=jnp.min(rendered), max=jnp.max(rendered[:,:,2])+0.1)
    viz = j.viz.resize_image(viz, image.intrinsics.height, image.intrinsics.width)
    viz.save(f"{RESULTS_IMGS_DIR}/dataset_{dataset_idx}_scene_{test_idx}_best_pred.png")

    return c2f_results
    

def test_dataset(dataset_idx, particles=1, save=True):
    data, num_scenes_in_dataset = load_dataset(dataset_idx)
    dataset_results = []

    for test_idx in range(num_scenes_in_dataset):
        print(f"#### Starting scene {test_idx}/{num_scenes_in_dataset} ####")

        c2f_results = test_scene(dataset_idx, test_idx, particles=particles)
        dataset_results.append(c2f_results)

    if save:
        ## Save whole dataset results
        # [[num_c2f_steps] * num_scenes_in_dataset]
        c2f_pred_weights = [[r[0][0] for r in c2f_stage_results] for c2f_stage_results in dataset_results]
        c2f_pred_poses = [[r[0][2] for r in c2f_stage_results] for c2f_stage_results in dataset_results]

        save_dataset_results(dataset_idx, c2f_pred_weights, c2f_pred_poses)

    return dataset_results


if __name__=='__main__':
    ######################
    # Single-object (of known identity) scenes
    # Infer the full pose in the full orientation space
    ######################
    RESULTS_DIR = 'c2f_results'
    RESULTS_DATA_DIR = f'{RESULTS_DIR}/data'
    RESULTS_IMGS_DIR = f'{RESULTS_DIR}/imgs'
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        os.makedirs(RESULTS_DATA_DIR)
        os.makedirs(RESULTS_IMGS_DIR)

    NUM_DATASET = 100
    NUM_PARTICLES = 5

    # setup renderer
    start = time.time() 
    renderer = setup_renderer(25, 0.3)
    end = time.time()
    print("setup renderer ", end-start)

    # configure c2f
    grid_widths = [0.1, 0.05, 0.03, 0.01]
    rot_angle_widths = [jnp.pi, jnp.pi, jnp.pi/1.5, jnp.pi/2.5]
    sphere_angle_widths = [jnp.pi, jnp.pi/6, jnp.pi/7, jnp.pi/8]
    grid_params =  [(3,3,3,75*5,61), (3,3,3,45,21),(3,3,3,45,21),(3,3,3,45,21)]  # (num_x, num_y, num_z, num_fib_sphere, num_planar_angle)
    scheds = j.c2f.make_schedules(
        grid_widths=grid_widths, 
        angle_widths=rot_angle_widths, 
        grid_params=grid_params, 
        full_pose=True, 
        sphere_angle_widths=sphere_angle_widths
    )

    # run test
    for dataset_idx in range(NUM_DATASET):
        c2f_results = test_dataset(dataset_idx, particles=NUM_PARTICLES)
    
    # c2f_results = test_scene(0, 2, particles=NUM_PARTICLES)
    # # c2f_results = test_scene(0, 4, particles=NUM_PARTICLES)


    

    from IPython import embed; embed()