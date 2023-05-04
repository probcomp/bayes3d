import jax.numpy as jnp
import bayes3d as j
import trimesh
import os
import numpy as np
import trimesh
import jax
import matplotlib.pyplot as plt
import time

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


intrinsics = j.Intrinsics(
    height=300,
    width=300,
    fx=400.0, fy=400.0,
    cx=150.0, cy=150.0,
    near=0.001, far=1.0
)

object_pose = j.distributions.gaussian_vmf_sample(
    jax.random.PRNGKey(4), 
    j.t3d.transform_from_pos(jnp.array([0.0, 0.0, 0.2])),
    0.00001, 0.001
)

object_pose_offset = (
    object_pose  @ j.t3d.inverse_pose(offset_pose)
)



rgb, seg, depth = j.kubric_interface.render_multiobject(
    [mesh_path],
    object_pose_offset[None, ...],
    jnp.eye(4),
    intrinsics,
    scaling_factor=1.0,
    lighting=5.0
)

_rgb = np.array(rgb)
_rgb[seg == 0, -1] = 0.0
_rgb[seg == 0, :] = 255.0
_depth = np.array(depth)
_depth[seg == 0] = 0.0

j.hstack_images(
    [
        j.get_rgb_image(_rgb),
        j.get_depth_image(
            _depth,max=intrinsics.far,
        ),
    ]
).save("data.png")

nominal_pose = j.t3d.transform_from_rot_and_pos(
    j.t3d.rotation_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.pi/2),
    jnp.array([0.0, 0.0, 0.2])
)
rgb_nominal, seg_nominal, depth_nominal = j.kubric_interface.render_multiobject(
    [mesh_path],
    nominal_pose[None, ...],
    jnp.eye(4),
    intrinsics,
    scaling_factor=1.0,
    lighting=5.0
)
_rgb_nominal = np.array(rgb_nominal)
_rgb_nominal[seg_nominal == 0, -1] = 0.0
j.get_rgb_image(_rgb_nominal).save("rgb_nominal.png")
_depth_nominal = np.array(depth_nominal)
_depth_nominal[seg_nominal == 0] = 0.0
j.get_depth_image(
    _depth_nominal,max=intrinsics.far,
).save("depth_nominal_.png")
depth_rgb = np.array(j.get_depth_image(
    _depth_nominal,max=intrinsics.far,
))
depth_rgb[seg_nominal == 0] = 0.0
j.get_rgb_image(depth_rgb,max=255.0).save("depth_nominal.png")

from IPython import embed; embed()



scaled_down_intrinsics = j.camera.scale_camera_parameters(intrinsics, 0.3333)
renderer = j.Renderer(scaled_down_intrinsics)
model_dir = os.path.join(j.utils.get_assets_dir(), "bop/ycbv/models")

mesh_path_ply = os.path.join(model_dir,"obj_" + "{}".format(IDX+1).rjust(6, '0') + ".ply")

renderer.add_mesh_from_file(
    mesh_path_ply
    ,scaling_factor=1.0/1000.0
    # os.path.join(model_dir,"obj_" + "{}".format(IDX).rjust(6, '0') + ".ply"), scaling_factor=1.0/1000.0
)
rerendered_img = renderer.render_single_object(
    object_pose,
    0
)
j.get_depth_image(rerendered_img[:,:,2], max=intrinsics.far).save("1.png")


NUM_POSES = 2048
keys = jax.random.split(jax.random.PRNGKey(5), NUM_POSES)
object_poses = jax.vmap(lambda key: j.distributions.gaussian_vmf_sample(
    key, object_pose, 0.00001, 1.0))(
    keys
)

start = time.time()
many_imgs = renderer.render_parallel(
    object_poses,
    0
)
end = time.time()
print(end-start)



r_sweep = jnp.array([0.01])
outlier_prob=0.1
outlier_volume=1.0


NUM_POSES = 2048
keys = jax.random.split(jax.random.PRNGKey(5), NUM_POSES)
object_poses = jax.vmap(lambda key: j.distributions.gaussian_vmf_sample(
    key, object_pose, 0.00001, 1.0))(
    keys
)

start = time.time()
many_imgs = renderer.render_parallel(object_poses,0)
print(many_imgs[0,0,0])
end = time.time()
print(end-start)

start = time.time()
weights = j.threedp3_likelihood_with_r_parallel_jit(
    point_cloud_image,
    many_imgs,
    r_sweep,
    outlier_prob,
    outlier_volume,
)
print(weights[0,0])
end = time.time()
print(end-start)

start = time.time()
many_imgs = renderer.render_parallel(object_poses,0)
weights = j.threedp3_likelihood_with_r_parallel_jit(
    point_cloud_image,
    many_imgs,
    r_sweep,
    outlier_prob,
    outlier_volume,
)
print(weights.argmax())
end = time.time()
print(end-start)

point_cloud_image = j.t3d.unproject_depth(j.utils.resize(depth, 
    scaled_down_intrinsics.height, scaled_down_intrinsics.width), scaled_down_intrinsics)
weights = j.threedp3_likelihood_with_r_parallel_jit(
    point_cloud_image,
    many_imgs,
    r_sweep,
    outlier_prob,
    outlier_volume,
)
best_img = many_imgs[weights.argmax()]
j.get_depth_image(best_img[:,:,2], max=intrinsics.far).save("best.png")





### Pose Inference





from IPython import embed; embed()