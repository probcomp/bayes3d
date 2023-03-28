import numpy as np
import os
import jax
import jax.numpy as jnp
import trimesh
import time
import pickle
import jax3dp3.transforms_3d as t3d
import jax3dp3 as j

from c2f_test_utils import *

DATASET_FILE = "rgbd_annotated.npz"  # npz file


# TODO: move these to utils
def load_data(scene_idx):
    scenes_data = np.load(DATASET_FILE, allow_pickle=True)
    rgbd_img, gt_idx, gt_pose = scenes_data['rgbd_idx_pose'][scene_idx]

    # filter 
    rgbd_img.rgb[rgbd_img.segmentation == 0,:] = 255.0
    rgbd_img.depth[rgbd_img.segmentation == 0] = 10.0
    
    # sanity check
    if gt_idx not in rgbd_img.segmentation:
        raise ValueError("single object dataset should contain RGBD objects with gt object id in segmentation")
    
    return (rgbd_img, gt_idx, gt_pose)

def setup_renderer():
    sample_rgbd, _, _ = load_data(0) 

    ## setup intrinsics and renderer
    intrinsics = j.camera.scale_camera_parameters(sample_rgbd.intrinsics, 0.3)
    renderer = j.Renderer(intrinsics, num_layers=100)

    ## load models
    model_dir = os.path.join(j.utils.get_assets_dir(), "ycb_video_models/models")
    mesh_paths = []
    model_names = j.ycb_loader.MODEL_NAMES
    for name in model_names:
        mesh_path = os.path.join(model_dir,name,"textured.obj")
        mesh_paths.append(mesh_path)
        model_scaling_factor = 1.0
        renderer.add_mesh_from_file(
            mesh_path,
            scaling_factor=model_scaling_factor
        )

    print("finished renderer setup")
    return renderer


if __name__=='__main__':
    ######################
    # Single-object (of known identity) scenes
    # Infer the full pose in the full orientation space
    ######################
    
    # setup renderer
    renderer = setup_renderer()
    
    # configure c2f
    grid_widths = [0.05, 0.03, 0.02, 0.02]
    rot_angle_widths = [jnp.pi, jnp.pi, 0.001, jnp.pi / 10]
    sphere_angle_widths = [jnp.pi, jnp.pi, jnp.pi / 3, jnp.pi / 4]
    # grid_params = [(7,7,7,50,21), (7,7,7,50,21), (15,15,15,20,1), (7,7,7,10,21)]   # (num_x, num_y, num_z, num_fib_sphere, num_planar_angle) 
    grid_params = [(5,5,5,5,5), (5,5,5,5,5),(5,5,5,5,5),(5,5,5,5,5)]    

    scheds = j.c2f.make_schedules(
        grid_widths=grid_widths, 
        angle_widths=rot_angle_widths, 
        grid_params=grid_params, 
        full_pose=True, 
        sphere_angle_widths=sphere_angle_widths
    )

    # choose testing scene
    test_idx = 0
    image, gt_idx, gt_pose = load_data(test_idx)
    j.viz.get_depth_image(image.depth).save("gt.png")

    seg_id = 0  # single object scene  # TODO check that kubric seg is 0 for obj
    c2f_results, viz = run_c2f(renderer, image, scheds, infer_id=False, infer_contact=False, viz=True)


    from IPython import embed; embed()