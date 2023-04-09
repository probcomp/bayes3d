import numpy as np
import os
import jax
import jax.numpy as jnp
import trimesh
import time
import pickle
import jax3dp3.transforms_3d as t3d
import jax3dp3 as j

from c2f_scripts import *

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
    renderer = j.Renderer(intrinsics, num_layers=25)

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


if __name__=='__main__':
    ######################
    # Single-object (of known identity) scenes
    # Infer the full pose in the full orientation space
    ######################
    
    # setup renderer
    start = time.time() 
    renderer = setup_renderer()
    end = time.time()
    print("that took ", end-start)

    # configure c2f
    grid_widths = [0.1, 0.05, 0.03, 0.01, 0.01, 0.01]
    rot_angle_widths = [jnp.pi, jnp.pi, jnp.pi, jnp.pi, jnp.pi/5, jnp.pi/5]
    sphere_angle_widths = [jnp.pi, jnp.pi/2, jnp.pi/4, jnp.pi/4, jnp.pi/5, jnp.pi/5]
    grid_params =  [(3,3,3,75*5,15), (3,3,3,75*3,21),(3,3,3,55,45),(3,3,3,55,45), (3,3,3,45,45), (3,3,3,45,45)]  # (num_x, num_y, num_z, num_fib_sphere, num_planar_angle)


    scheds = j.c2f.make_schedules(
        grid_widths=grid_widths, 
        angle_widths=rot_angle_widths, 
        grid_params=grid_params, 
        full_pose=True, 
        sphere_angle_widths=sphere_angle_widths
    )

    # choose testing scene
    for test_idx in [0,1,2,3]:
    # test_idx = 1
        image, gt_idx, gt_pose = load_data(test_idx)
        j.viz.get_depth_image(image.depth).save(f"scene_{test_idx}_gt.png")
        rendered = renderer.render_single_object(gt_pose, gt_idx)  
        rendered -= jnp.min(rendered)  # to make minimum 0
        viz = j.viz.get_depth_image(rendered[:,:,2], max=jnp.max(rendered[:,:,2])+0.5)
        viz = j.viz.resize_image(viz, image.intrinsics.height, image.intrinsics.width)
        viz.save(f"scene_{test_idx}_gt_depth.png")

        seg_id = 0  # single object scene  # TODO check that kubric seg is 0 for obj
        c2f_results = run_c2f(renderer, image, scheds, infer_id=False, infer_contact=False, viz=True)
        c2f_results = c2f_results[0]  # single-segment, single-object
        
        print("gt:", gt_pose)
        for c2f_iter, c2f_result in enumerate(c2f_results):
            score, gt_idx, best_pose = c2f_result[0]
            print(best_pose)
            rendered = renderer.render_single_object(best_pose, gt_idx)  
            viz = j.viz.get_depth_image(rendered[:,:,2], min=jnp.min(rendered), max=jnp.max(rendered[:,:,2])+0.5)
            viz = j.viz.resize_image(viz, image.intrinsics.height, image.intrinsics.width)
            viz.save(f"scene_{test_idx}_best_pred_{c2f_iter}.png")
        
        print(f"=================\n test idx {test_idx} complete \n================")


    from IPython import embed; embed()