import jax.numpy as jnp
import jax3dp3 as j
import trimesh
import os
import numpy as np
import trimesh
import jax
import matplotlib.pyplot as plt
import time
import open3d as o3d


# --- creating the model dir from the working directory
model_dir = os.path.join(j.utils.get_assets_dir(), "ycb_video_models/models")
print(f"{model_dir} exists: {os.path.exists(model_dir)}")
mesh_paths = []
model_names = np.array(j.ycb_loader.MODEL_NAMES)
mesh_paths = []
for name in model_names:
    mesh_paths.append(os.path.join(model_dir,name,"textured.obj"))

IDX = 2
mesh_path = mesh_paths[IDX]
_, offset_pose = j.mesh.center_mesh(trimesh.load(mesh_path), return_pose=True)


intrinsics = j.Intrinsics(
    height=300,
    width=300,
    fx=800.0, fy=800.0,
    cx=150.0, cy=150.0,
    near=0.001, far=2.0
)


renderer = j.Renderer(intrinsics)
renderer.add_mesh_from_file(mesh_paths[IDX])

object_pose = j.distributions.gaussian_vmf_sample(
    jax.random.PRNGKey(6), 
    j.t3d.transform_from_pos(jnp.array([0.0, 0.0, 0.8])),
    0.00001, 0.001
)

img = renderer.render_single_object(
    object_pose @ j.t3d.inverse_pose(offset_pose),
    0
)
point_cloud = img[img[:,:,2] > 0.0,:3].reshape(-1,3)

noise = jax.vmap(
    lambda key: jax.random.multivariate_normal(
        key, jnp.zeros(3), jnp.eye(3)*0.0001
    )
)(
    jax.random.split(jax.random.PRNGKey(3), point_cloud.shape[0])
)
point_cloud_noisy = noise + point_cloud
img_noisy = j.render_point_cloud(point_cloud_noisy, intrinsics)
j.get_depth_image(img_noisy[:,:,2],max=intrinsics.far).save("noisy.png")



camera_pose = j.t3d.transform_from_pos_target_up(
    jnp.array([0.7, 0.0, 0.1]),
    jnp.array([0.0, 0.0, 0.0]),
    jnp.array([0.0, 0.0, 1.0]),
)

object_pose = j.t3d.inverse_pose(camera_pose) @ j.t3d.inverse_pose(offset_pose)

images = []

for IDX in range(6):
    rgb, seg, depth = j.kubric_interface.render_multiobject(
        [mesh_paths[IDX]],
        object_pose[None, ...],
        jnp.eye(4),
        intrinsics,
        scaling_factor=1.0,
        lighting=3.0
    )

    _rgb = np.array(rgb)
    _rgb[seg == 0, -1] = 0.0
    _rgb[seg == 0, :] = 255.0
    images.append(j.get_rgb_image(_rgb))

j.hstack_images(images).save("images.png")






r = 0.01
noise = jax.vmap(
    lambda key: jax.random.multivariate_normal(
        key, jnp.zeros(3), jnp.eye(3)*r
    )
)(
    jax.random.split(jax.random.PRNGKey(3), point_cloud.shape[0])
)
point_cloud_noisy = noise + point_cloud

img_noisy = j.  (point_cloud_noisy, intrinsics)

viz = j.o3d_viz.O3DVis(intrinsics)
mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=np.array([0.0, 0.0, 0.0]))
mesh.transform(np.array(object_pose))
mtl = o3d.visualization.rendering.MaterialRecord()  # or MaterialRecord(), for later versions of Open3D
mtl.shader = "defaultUnlit"
viz.clear()
viz.render.scene.add_geometry(f"1", mesh, mtl)

img = np.array(viz.capture_image(intrinsics, np.eye(4)))
img[img[:,:,:3].sum(-1)>220*3,:] = 0.0
j.get_rgb_image(img).save("o3d.png")


