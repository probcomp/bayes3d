import os 
import jax3dp3 as j
import jax 
import jax.numpy as jnp 
import numpy as np
import subprocess

COSYPOSE_CONDA_ENV_NAME = 'cosypose'

def setup_renderer(data):
    renderer = j.Renderer(data['rgbds'][0].intrinsics, num_layers=25)
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
    return renderer

def main():
    ## get data to predict
    DATASET_FILE = os.path.join(j.utils.get_assets_dir(), f"datasets/dataset_0.npz")
    data = np.load(DATASET_FILE, allow_pickle=True)
    rgbds = data['rgbds']
    gt_poses = data['poses']
    GT_NAME = str(data['name'])
    GT_IDX = int(data['id'])

    ## set up renderer
    renderer = setup_renderer(data)
    
    # multi-image inference
    TEST_IDXS = [i for i in range(len(rgbds))]    # list of all img idxs to test
    gt_poses = np.asarray(gt_poses[TEST_IDXS])
    rgb_imgs = np.asarray([rgbd.rgb[:,:,:3] for rgbd in rgbds[TEST_IDXS]]) 
    intrinsics = rgbds[0].intrinsics

    preds = j.cosypose_utils.cosypose_interface(rgb_imgs, j.K_from_intrinsics(intrinsics))  # poses,poses_input,K_crop,boxes_rend,boxes_crop

    print("num of pred:",len(preds))

    pred_poses, pred_ids, pred_scores = preds['pred_poses'], preds['pred_ids'], preds['pred_scores']
    print(f"predicted ID {pred_ids}, score {pred_scores}")

    # render gt, pred
    gt_vizs_list, pred_vizs_list = [], []
    gt_renders = renderer.render_parallel(jnp.asarray(gt_poses), GT_IDX)  
    pred_renders = renderer.render_parallel(jnp.asarray(pred_poses), [pred_id[0] for pred_id in pred_ids])  
    for gt_render, pred_render in zip(gt_renders, pred_renders):
        gt_viz = j.viz.get_depth_image(gt_render[:,:,2], min=jnp.min(gt_render), max=5.0)
        pred_viz = j.viz.get_depth_image(pred_render[:,:,2], min=jnp.min(pred_render), max=5.0)

        gt_vizs_list.append(gt_viz)
        pred_vizs_list.append(pred_viz)
    gt_viz, pred_vizs = j.hstack_images(gt_vizs_list), j.hstack_images(pred_vizs_list)

    comparison_viz = j.vstack_images([gt_viz, pred_vizs])
    comparison_viz.save(f"comparison_{GT_IDX}_{GT_NAME}.png")

    from IPython import embed; embed()


if __name__ == '__main__':
    main()

