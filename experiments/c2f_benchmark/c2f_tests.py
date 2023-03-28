import numpy as np
import os
import jax
import jax.numpy as jnp
import trimesh
import time
import pickle
import jax3dp3.transforms_3d as t3d
import jax3dp3

from c2f_test_utils import *


from IPython import embed; embed()
import argparse

parser = argparse.ArgumentParser(description='Specify testing mode')
parser.add_argument('--test_mode', dest='TEST_MODE', type=str, default='c2f')   # c2f or grid
parser.add_argument('--data_mode', dest='DATA_MODE', type=str, default='obj')   # obj or ply


args = parser.parse_args()

DATA_MODE = args.DATA_MODE
TEST_MODE = args.TEST_MODE

data_modes = ['obj', 'ply']
test_modes = ['c2f','grid']
if DATA_MODE not in data_modes or TEST_MODE not in test_modes:
    raise ValueError("Invalid arguments")

print("Data mode ", DATA_MODE)
print("Test mode ", TEST_MODE)


#########
# TESTING FCNS (TODO move to file)
#########

def viz_results(results, tested_likelihoods,gridding_idx=0):    

    if pred_obj_idx == gt_obj_idx:
        accuracy = 'correct'
        print(f"{TEST_MODE} {save_filename} Classification {accuracy}: pred,gt {model_names[pred_obj_idx]}")
    else:
        accuracy = 'INCORRECT'
        print(f"{TEST_MODE} {save_filename} Classification {accuracy}: pred {model_names[pred_obj_idx]} for gt {model_names[gt_obj_idx]}")

    panel_viz = jax3dp3.c2f.multi_panel_c2f_viz(
    results, rgb_original, gt_image_masked, h, w, far, 
    model_names, title=f"Likelihoods: {tested_likelihoods}, Outlier Params: {outlier_prob},{outlier_volume}")

    panel_viz.save(f"{save_img_dir}/{gt_obj_name}/{TEST_MODE}/{save_filename}_grid_{gridding_idx}_id_{seg_id}_{accuracy}.png")
    # jax3dp3.viz.multi_panel_vertical(all_panels_for_segmentation).save(f"imgs/{save_filename}_id_{seg_id}_all_steps.png")

    print("\n")

def save_results():
    with open(f"{save_data_dir}/{gt_obj_name}/{TEST_MODE}/{DATA_MODE}_{save_filename}_id_{seg_id}.pkl", 'wb') as file:
        pickle.dump(results_dict, file)
    

############
# Process data
############

TEST_OBJ_NAMES = [
    "003_cracker_box",      
    "004_sugar_box",        
    "005_tomato_soup_can",  
    "010_potted_meat_can",   
    "011_banana",         
    "019_pitcher_base",      
    "021_bleach_cleanser",   
    "035_power_drill", 
    "036_wood_block",        
    "040_large_marker",   
    "051_large_clamp",   
    "052_extra_large_clamp"
]
 
load_data_dir = "/home/ubuntu/jax3dp3/experiments/c2f_benchmark/test_data/data"


np.random.seed(0)
# num_trials_per_scene = 20  # choose random trials
for test_obj_name in TEST_OBJ_NAMES:
    print("======================")
    print(test_obj_name)
    print("======================")


    num_tests = len(os.listdir(os.path.join(load_data_dir,test_obj_name)))
    num_trials_per_scene = num_tests
    for ii in range(num_trials_per_scene):
        obj_test_idx = np.random.randint(num_tests)

        filename = f"data_{obj_test_idx}.pkl"
        save_filename = filename.split('.')[0].split('/')[-1]
        print(f"\nProcessing scene {filename}...")

        with open(os.path.join(load_data_dir, os.path.join(test_obj_name,filename)),'rb') as file:
            data = pickle.load(file)
            
        rgb_original = data["rgb"]
        depth_original = data["depth"] / data["factor_depth"]
        seg_original = data['segmentation']
        K = data["intrinsics"]
        cam_pose_original = data['camera_pose']
        object_pose_original = data['object_pose']
        contact_param = data['contact_param']
        contact_face = data['contact_face']
        contact_plane = data['contact_plane']
        gt_obj_name = data['object_name']


        #########
        #  Setup intrinsics and renderer
        #########
        orig_h,orig_w = depth_original.shape

        orig_fx, orig_fy, orig_cx, orig_cy = K[0,0],K[1,1],K[0,2],K[1,2]
        near,far = 0.001, 5.0 #20.0

        h,w,fx,fy,cx,cy = jax3dp3.camera.scale_camera_parameters(orig_h,orig_w,orig_fx,orig_fy,orig_cx,orig_cy, scaling_factor=0.25)
        depth = jax3dp3.utils.resize(depth_original, h, w)
        depth[depth > far] = 0.0

        gt_image_full = gt_image_above_table = t3d.depth_to_point_cloud_image(depth, fx, fy, cx, cy)
        gt_point_cloud_full = t3d.point_cloud_image_to_points(gt_image_full)

        table_dims = [10,10,1e-10]
        table_face_param = 2
        table_surface_plane_pose =jnp.eye(4)
        cam_pose = cam_pose_original


        #############
        # Load models
        #############

        model_dir_ply = os.path.join(jax3dp3.utils.get_assets_dir(), "bop/ycbv/models")
        model_dir_obj = os.path.join(jax3dp3.utils.get_assets_dir(), "ycb_obj/models")
        model_names = jax3dp3.ycb_loader.MODEL_NAMES
        model_paths = []
        for idx in range(21):
            if DATA_MODE == 'obj':    
                model_paths.append(os.path.join(model_dir_obj, os.path.join(model_names[idx], 'textured_simple.obj')))
                model_scaling_factor = 1.0
            elif DATA_MODE == 'ply': 
                model_paths.append(os.path.join(model_dir_ply,"obj_" + f"{str(idx+1).rjust(6, '0')}.ply"))
                model_scaling_factor = 1.0/1000.0
        # print(model_paths)

        jax3dp3.setup_renderer(h, w, fx, fy, cx, cy, near, far, num_layers=100)

        model_box_dims = []
        for path in model_paths:
            mesh = trimesh.load(path)  # 000001 to 000021
            mesh.vertices = mesh.vertices * model_scaling_factor
            mesh = jax3dp3.mesh.center_mesh(mesh)

            model_box_dims.append(jax3dp3.utils.aabb(mesh.vertices)[0])
            jax3dp3.load_model(mesh)
        model_box_dims = jnp.array(model_box_dims)


        ## Check data so far; generate dashboard    
        segmentation_img = jnp.asarray(jax3dp3.utils.resize(seg_original, h,w))
        unique =  np.unique(segmentation_img)
        print(unique)
        gt_obj_idx = model_names.index(gt_obj_name)  #Render gt object at gt pose
        gt_obj_pose = t3d.inverse_pose(cam_pose_original) @ object_pose_original
        gt_pose_render = jax3dp3.renderer.render_single_object(jnp.asarray(gt_obj_pose), gt_obj_idx)

        if not os.path.exists(f"imgs/{gt_obj_name}"):
            os.mkdir(f"imgs/{gt_obj_name}")
        jax3dp3.viz.multi_panel(
            [
                jax3dp3.viz.resize_image(jax3dp3.viz.get_rgb_image(rgb_original, 255.0),h,w),
                jax3dp3.viz.get_depth_image(gt_image_full[:,:,2],  max=far),
                jax3dp3.viz.get_depth_image(gt_image_above_table[:,:,2],  max=far),
                jax3dp3.viz.get_depth_image(segmentation_img + 1, max=segmentation_img.max() + 1),
                jax3dp3.viz.get_depth_image(gt_pose_render[:,:,2], min=near,max=far)
            ],
            labels=["RGB", "Depth", "Above Table", "Segmentation", "GT Pose Render"],
            bottom_text="Intrinsics {:d} {:d} {:0.2f} {:0.2f} {:0.2f} {:0.2f} {:0.2f} {:0.2f}\n".format(h,w,fx,fy,cx,cy,near,far),
        ).save(f"imgs/{gt_obj_name}/{DATA_MODE}_{obj_test_idx}_dashboard.png")


        ##########
        # RUN TEST (C2F or GRID)
        ##########

        outlier_prob=0.08
        outlier_volume=10**3

        c2f_grid_widths=[0.08, 0.07, 0.04, 0.02]
        c2f_grid_params = [(5,5,7),(5,5,5),(5,5,5),(5,5,3)]   # TODO 
        c2f_grid_params_const_angle = [(5,5,5),(5,5,10),(5,5,10),(5,5,10)]   # TODO 

        gridding_grid_widths, gridding_grid_params = get_gridding_cfg_from_c2f_sched(c2f_grid_widths, c2f_grid_params_const_angle)

        likelihood_r_sched = np.array([0.08, 0.04, 0.02, 0.02])

        print(outlier_prob, likelihood_r_sched)

        if TEST_MODE == 'grid':
            contact_param_sched, face_param_sched = jax3dp3.c2f.make_schedules(
                grid_widths=gridding_grid_widths, grid_params=gridding_grid_params
            )
            likelihood_r_sched = [likelihood_r_sched[-1]] * len(likelihood_r_sched)  # single likeoihood
        elif TEST_MODE == 'c2f':
            contact_param_sched, face_param_sched = jax3dp3.c2f.make_schedules_reduce_angle(
                grid_widths=c2f_grid_widths, grid_params=c2f_grid_params
            )

        # visualize
        save_img_dir = "eval_results/imgs"
        save_data_dir = "eval_results/data"

        if not os.path.exists(f"{save_img_dir}"):
            os.mkdir(f"{save_img_dir}")
        if not os.path.exists(f"{save_data_dir}"):
            os.mkdir(f"{save_data_dir}")

        if not os.path.exists(f"{save_img_dir}/{gt_obj_name}"):
            os.mkdir(f"{save_img_dir}/{gt_obj_name}")
        if not os.path.exists(f"{save_img_dir}/{gt_obj_name}/{TEST_MODE}"):
            os.mkdir(f"{save_img_dir}/{gt_obj_name}/{TEST_MODE}")
        if not os.path.exists(f"{save_data_dir}/{gt_obj_name}"):
            os.mkdir(f"{save_data_dir}/{gt_obj_name}")
        if not os.path.exists(f"{save_data_dir}/{gt_obj_name}/{TEST_MODE}"):
            os.mkdir(f"{save_data_dir}/{gt_obj_name}/{TEST_MODE}")

        for seg_id in np.unique(segmentation_img):
            if seg_id == -1:
                continue

            print("\n\nSegment ID: ", seg_id)

            scheds = [contact_param_sched, face_param_sched, likelihood_r_sched]

            if TEST_MODE == 'c2f':
                c2f_results, gt_image_masked, elapsed = get_c2f_results(seg_id, scheds, r_final = 0.02, viz=True)
                    # all_panels_for_segmentation.append(panel_viz)
                top_result = c2f_results[0]
                _, _, pred_pose, pred_obj_idx, pred_img, _, _ = top_result

                results_dict = {'time': elapsed,  # point cloud image; see t3d.point_cloud_image_to_points
                    'contact_param_sched':contact_param_sched,
                    'face_param_sched': face_param_sched,
                    'likelihood_r_sched':np.asarray(likelihood_r_sched),
                    # 'gt_img':gt_image_full,  # point cloud image 
                    # 'c2f_pred_img':pred_img,
                    'intrinsics':np.array([[fx,0,cx],[0,fy,cy],[0,0,1]]),
                    'gt_pose':gt_obj_pose,
                    'pred_pose':pred_pose,
                    'gt_obj_name':gt_obj_name,
                    'gt_obj_idx':gt_obj_idx,
                    'pred_obj_idx':pred_obj_idx
                    }
                viz_results(c2f_results, likelihood_r_sched)
                save_results()

            elif TEST_MODE == 'grid':

                num_grids = len(scheds[0])  # test each of the gridding parameters in the c2f schedule
                pred_poses = []
                pred_obj_idxs = []
                

                for i_grid in range(num_grids):
                    sched = [contact_param_sched[i_grid], face_param_sched[i_grid], likelihood_r_sched[i_grid]]
                    print(f"Grid test {i_grid}: sched {gridding_grid_params[i_grid]}")
                    grid_results, gt_image_masked, elapsed = get_grid_results(seg_id, sched, viz=True)
                    # all_panels_for_segmentation.append(panel_viz)
                    top_result = grid_results[0]
                    _, _, pred_pose, pred_obj_idx, pred_img, _, _ = top_result

                    pred_poses.append(pred_pose)
                    pred_obj_idxs.append(pred_obj_idx)

                    viz_results(grid_results, [likelihood_r_sched[i_grid]],gridding_idx=i_grid)


                # from IPython import embed; embed()
                results_dict = {'time': elapsed,  # point cloud image; see t3d.point_cloud_image_to_points
                    'contact_param_sched':np.asarray(contact_param_sched),  # order in which the grid params are processed
                    'face_param_sched': np.asarray(face_param_sched),
                    'likelihood_r_sched':np.asarray(likelihood_r_sched),
                    'intrinsics':np.array([[fx,0,cx],[0,fy,cy],[0,0,1]]),
                    'gt_pose':gt_obj_pose,
                    'pred_poses':np.asarray(pred_poses),
                    'gt_obj_name':gt_obj_name,
                    'gt_obj_idx':gt_obj_idx,
                    'pred_obj_idxs':np.asarray(pred_obj_idxs)
                    }     
                save_results()  # save all gridding results for the test file into one pik

            


from IPython import embed; embed()


