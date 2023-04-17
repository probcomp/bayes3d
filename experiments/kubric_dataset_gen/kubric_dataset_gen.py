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
model_names = j.ycb_loader.MODEL_NAMES
IDX = 13
name = model_names[IDX]
print(name)

camera_pose = j.t3d.transform_from_pos_target_up(
    jnp.array([0.0, 0.5, 0.0]),
    jnp.array([0.0, 0.0, 0.0]),
    jnp.array([0.0, 0.0, 1.0]),
)


bop_ycb_dir = os.path.join(j.utils.get_assets_dir(), "bop/ycbv")
rgbd, gt_ids, gt_poses, masks = j.ycb_loader.get_test_img('52', '1', bop_ycb_dir)
intrinsics = j.Intrinsics(
    height=rgbd.intrinsics.height,
    width=rgbd.intrinsics.width,
    fx=rgbd.intrinsics.fx, fy=rgbd.intrinsics.fx,
    cx=rgbd.intrinsics.width/2.0, cy=rgbd.intrinsics.height/2.0,
    near=0.001, far=3.0
)




NUM_IMAGES_PER_ITER = 10
NUM_ITER = 100

for iter in range(NUM_ITER):
    print("Iteration: ", iter)
    key = jax.random.PRNGKey(iter)
    object_poses = jax.vmap(lambda key: j.distributions.gaussian_vmf(key, 0.00001, 0.001))(
        jax.random.split(key, NUM_IMAGES_PER_ITER)
    )
    object_poses = jnp.einsum("ij,ajk",j.t3d.inverse_pose(camera_pose),object_poses)

    mesh_paths = []
    mesh_path = os.path.join(model_dir,name,"textured.obj")
    for _ in range(NUM_IMAGES_PER_ITER):
        mesh_paths.append(mesh_path)
    _, offset_pose = j.mesh.center_mesh(trimesh.load(mesh_path), return_pose=True)


    all_data = j.kubric_interface.render_multiobject_parallel(mesh_paths, object_poses[None,:,...], intrinsics, scaling_factor=1.0, lighting=3.0) # multi img singleobj
    gt_poses = object_poses @ offset_pose

    DATASET_FILENAME = f"dataset_{iter}.npz"  # npz file
    DATASET_FILE = os.path.join(j.utils.get_assets_dir(), f"datasets/{DATASET_FILENAME}")
    np.savez(DATASET_FILE, rgbds=all_data, poses=gt_poses, id=IDX, name=model_names[IDX], intrinsics=intrinsics, mesh_path=mesh_path)

    rgb_images = j.hstack_images([j.get_rgb_image(r.rgb) for r in all_data]).save(f"dataset_{iter}.png")

from IPython import embed; embed()