import jax.numpy as jnp
import jax3dp3 as j
import trimesh
import os
import numpy as np
import trimesh
import jax


NUM_IMAGES = 4
# --- creating the model dir from the working directory
model_dir = os.path.join(j.utils.get_assets_dir(), "ycb_video_models/models")
print(f"{model_dir} exists: {os.path.exists(model_dir)}")
mesh_paths = []
model_names = j.ycb_loader.MODEL_NAMES
IDX = 13
name = model_names[IDX]
print(name)
mesh_path = os.path.join(model_dir,name,"textured.obj")
for _ in range(NUM_IMAGES):
    mesh_paths.append(mesh_path)
_, offset_pose = j.mesh.center_mesh(trimesh.load(mesh_path), return_pose=True)

camera_pose = j.t3d.transform_from_pos_target_up(
    jnp.array([0.0, 0.8, 0.0]),
    jnp.array([0.0, 0.0, 0.0]),
    jnp.array([0.0, 0.0, 1.0]),
)

key = jax.random.PRNGKey(3)
object_poses = jax.vmap(lambda key: j.distributions.gaussian_vmf(key, 0.00001, 0.001))(
    jax.random.split(key, NUM_IMAGES)
)
object_poses = jnp.einsum("ij,ajk",j.t3d.inverse_pose(camera_pose),object_poses)



bop_ycb_dir = os.path.join(j.utils.get_assets_dir(), "bop/ycbv")
rgbd, gt_ids, gt_poses, masks = j.ycb_loader.get_test_img('52', '1', bop_ycb_dir)
intrinsics = j.Intrinsics(
    height=rgbd.intrinsics.height,
    width=rgbd.intrinsics.width,
    fx=rgbd.intrinsics.fx, fy=rgbd.intrinsics.fx,
    cx=rgbd.intrinsics.width/2.0, cy=rgbd.intrinsics.height/2.0,
    near=0.001, far=50.0
)

# all_data = j.kubric_interface.render_multiobject_parallel(mesh_paths, object_poses[:,None,...], intrinsics, scaling_factor=1.0, lighting=5.0) # single image multiobj
all_data = j.kubric_interface.render_multiobject_parallel(mesh_paths, object_poses[None,:,...], intrinsics, scaling_factor=1.0, lighting=5.0) # multi img singleobj

from IPython import embed; embed()

rgb_viz = []
for d in all_data:
    rgb = d.rgb
    seg = d.segmentation
    depth = d.depth

    ## visualize rgba
    rgba = jnp.array(j.viz.add_rgba_dimension(rgb))
    rgba = rgba.at[seg == 0, 3].set(0.0)

    # depth_viz = j.get_depth_image(depth, max=intrinsics.far)
    # seg_viz = j.get_depth_image(seg, max=seg.max())

    rgb_viz.append(j.get_rgb_image(rgba))

    ## modify segmentation to reflect gt object id
    unique_ids = np.unique(seg)
    unique_ids = unique_ids[unique_ids != 0]  
    assert len(unique_ids) == 1
    seg[seg == unique_ids[0]] = IDX 
    
    d.segmentation = seg

j.hstack_images(rgb_viz).save("dataset.png")

rgbs=np.array([i.rgb for i in all_data])
segs=np.array([i.segmentation for i in all_data])
depths=np.array([i.depth for i in all_data])
gt_idxs=np.array([IDX for _ in all_data])
gt_poses=np.array([object_poses])
annotated_data = np.array([(rgbd_data, IDX, object_pose) for rgbd_data, object_pose in zip(all_data, object_poses)], dtype=object)

# np.savez("mug_images.npz", rgbs=rgbs, depths=depths)
np.savez("rgbd_annotated.npz", rgbd_idx_pose=annotated_data)
np.savez("rgbd.npz", rgbd=all_data, gt_idxs=gt_idxs, gt_poses=gt_poses)

from IPython import embed; embed()
