import jax.numpy as jnp
import jax3dp3 as j
import trimesh
import os
import numpy as np
import trimesh
import jax

# --- creating the model dir from the working directory
model_dir = os.path.join(j.utils.get_assets_dir(), "ycb_video_models/models")
print(f"{model_dir} exists: {os.path.exists(model_dir)}")
mesh_paths = []
model_names = j.ycb_loader.MODEL_NAMES
IDX = 13
name = model_names[IDX]
print(name)
mesh_path = os.path.join(model_dir,name,"textured.obj")
_, offset_pose = j.mesh.center_mesh(trimesh.load(mesh_path), return_pose=True)

camera_pose = j.t3d.transform_from_pos_target_up(
    jnp.array([0.0, 0.8, 0.0]),
    jnp.array([0.0, 0.0, 0.0]),
    jnp.array([0.0, 0.0, 1.0]),
)

key = jax.random.PRNGKey(3)
object_poses = jax.vmap(lambda key: j.distributions.gaussian_vmf(key, 0.00001, 0.001))(
    jax.random.split(key, 1)
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

all_data = j.kubric_interface.render_parallel(mesh_path, object_poses, jnp.eye(4), intrinsics, scaling_factor=1.0, lighting=5.0)

rgb_viz = []
for d in all_data:
    rgb_viz.append(j.get_rgb_image(d[0]))

j.hstack_images(rgb_viz).save("dataset.png")

rgbs=np.array([i[0] for i in all_data])
segs=np.array([i[1] for i in all_data])
depths=np.array([i[2] for i in all_data])

np.savez("mug_images.npz", rgbs=rgbs, depths=depths)

from IPython import embed; embed()

import jax3dp3.posecnn_densefusion
densefusion = j.posecnn_densefusion.DenseFusion()

t = 0
rgb = rgbs[t]
seg = segs[t]


rgb[seg == 0,:] = 255.0
depth[seg ==0] = 10.0


j.get_rgb_image(rgb[:,:,:3]).save("img3.png")

results = densefusion.get_densefusion_results(rgb, depth, intrinsics, scene_name="1")
results = densefusion.get_densefusion_results(rgb, rgbd.depth, intrinsics, scene_name="1")
results = densefusion.get_densefusion_results(rgbd.rgb, rgbd.depth, intrinsics, scene_name="1")
results = densefusion.get_densefusion_results(rgbd.rgb, depth, intrinsics, scene_name="1")

j.meshcat.setup_visualizer()
j.meshcat.show_cloud("1", j.t3d.unproject_depth(rgbd.depth, intrinsics).reshape(-1,3))
j.meshcat.show_cloud("2", j.t3d.unproject_depth(depth, intrinsics).reshape(-1,3),color=j.RED)


from IPython import embed; embed()
