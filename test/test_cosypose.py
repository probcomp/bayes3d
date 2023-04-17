import os 
import jax3dp3 as j
import jax 
import jax.numpy as jnp 
import numpy as np
import subprocess

COSYPOSE_CONDA_ENV_NAME = 'cosypose'

def main():
    ## get data to predict
    DATASET_FILE = os.path.join(j.utils.get_assets_dir(), f"datasets/dataset_0.npz")
    data = np.load(DATASET_FILE, allow_pickle=True)
    rgbds = data['rgbds']
    gt_poses = data['poses']
    name = str(data['name'])
    GT_IDX = int(data['id'])

    ## set up renderer
    renderer = j.Renderer(rgbds[0].intrinsics, num_layers=25)
    # load models
    model_dir = os.path.join(j.utils.get_assets_dir(), "bop/ycbv/models")
    model_names = ["obj_" + f"{str(idx+1).rjust(6, '0')}.ply" for idx in range(14)]
    mesh_paths = []
    for name in model_names:
        mesh_path = os.path.join(model_dir,name)
        mesh_paths.append(mesh_path)
        model_scaling_factor = 1.0/1000.0
        renderer.add_mesh_from_file(
            mesh_path,
            scaling_factor=model_scaling_factor
        )

    for IMG_IDX in range(len(rgbds)):
        rgbd = rgbds[IMG_IDX]
        gt_pose = jnp.asarray(gt_poses[IMG_IDX])
        rgb_img = np.asarray(rgbd.rgb[:,:,:3]) 
        intrinsics = rgbd.intrinsics
        camera_k = j.K_from_intrinsics(intrinsics)

        ## run cosypose predictions
        #########
        # TODO run subprocssing once for entire dataset
        #########

        # pred = j.cosypose_utils.cosypose_interface(rgb_img,camera_k)
        pred = j.cosypose_utils.cosypose_interface(rgb_img,camera_k)

        #poses,poses_input,K_crop,boxes_rend,boxes_crop
        print("num of pred:",len(pred))

        pred_poses, pred_ids, pred_scores = pred['pred_poses'], pred['pred_ids'], pred['pred_scores']
        print(f"predicted ID {pred_ids}, score {pred_scores}")
        pred_pose = jnp.asarray(pred_poses[0])

        # render gt, pred
        rendered = renderer.render_single_object(gt_pose, GT_IDX)  
        viz = j.viz.get_depth_image(rendered[:,:,2], min=jnp.min(rendered), max=5.0)
        gt_viz = j.viz.resize_image(viz, intrinsics.height, intrinsics.width)

        rendered = renderer.render_single_object(pred_pose, GT_IDX)  
        viz = j.viz.get_depth_image(rendered[:,:,2], min=jnp.min(rendered), max=5.0)
        pred_viz = j.viz.resize_image(viz, intrinsics.height, intrinsics.width)

        comparison_viz = j.hstack_images([gt_viz, pred_viz])
        comparison_viz.save(f"comparison_{IMG_IDX}.png")

if __name__ == '__main__':
    main()
    from IPython import embed; embed()

