import os 
import jax3dp3 as j
import jax 
import jax.numpy as jnp 
import numpy as np
import subprocess

bop_ycb_dir = os.path.join(j.utils.get_assets_dir(), "bop/ycbv")
rgbd, gt_ids, gt_poses, masks = j.ycb_loader.get_test_img('52', '1', bop_ycb_dir)

pred = j.cosypose_utils.cosypose_interface(np.array(rgbd.rgb), j.K_from_intrinsics(rgbd.intrinsics))

pred_poses, pred_ids, pred_scores = pred['pred_poses'], pred['pred_ids'], pred['pred_scores']

renderer = j.Renderer(rgbd.intrinsics, num_layers=25)
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

    
rendered = renderer.render_multiobject(jnp.array(pred_poses), pred_ids)  
viz = j.multi_panel(
    [
        j.get_rgb_image(rgbd.rgb),
        j.get_depth_image(rgbd.depth),
        j.get_depth_image(rendered[:,:,2])
    ],
    labels=[
        "RGB",
        "Depth",
        "Reconstruction"
    ]
)
viz.save("test_cosypose.png")

