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
IDX = 1
name = model_names[IDX]
mesh_path = os.path.join(model_dir,name,"textured.obj")
_, offset_pose = j.mesh.center_mesh(trimesh.load(mesh_path), return_pose=True)

camera_pose = j.t3d.transform_from_pos_target_up(
    jnp.array([0.0, 0.4, 0.0]),
    jnp.array([0.0, 0.0, 0.0]),
    jnp.array([0.0, 0.0, 1.0]),
)

key = jax.random.PRNGKey(3)
object_poses = jax.vmap(lambda key: j.distributions.gaussian_vmf(key, 0.00001, 0.001))(
    jax.random.split(key, 10)
)
object_poses = jnp.einsum("ij,ajk",j.t3d.inverse_pose(camera_pose),object_poses)


intrinsics = j.Intrinsics(
    height=300,
    width=300,
    fx=200.0, fy=200.0,
    cx=150.0, cy=150.0,
    near=0.001, far=50.0
)

all_data = j.kubric_interface.render_parallel(mesh_path, object_poses, jnp.eye(4), intrinsics, scaling_factor=1.0, lighting=5.0)

from IPython import embed; embed()
rgb_viz = []
for d in all_data:
    rgb_viz.append(j.get_rgb_image(d[0]))

j.hstack_images(rgb_viz).save("dataset.png")