import os

import jax
import jax.numpy as jnp

import bayes3d as b

b.setup_visualizer()


N = 100
cloud = jax.random.uniform(jax.random.PRNGKey(10), shape=(N, 3)) * 0.1
b.show_cloud("c", cloud)


pose = b.distributions.gaussian_vmf_zero_mean(jax.random.PRNGKey(5), 0.1, 10.0)

cloud_transformed = b.apply_transform(cloud, pose)
b.show_cloud("d", cloud_transformed, color=b.RED)

transform = b.utils.find_least_squares_transform_between_clouds(
    cloud, cloud_transformed
)

print(jnp.abs(cloud - cloud_transformed).sum())
print(jnp.abs(cloud_transformed - b.apply_transform(cloud, transform)).sum())


intrinsics = b.Intrinsics(
    height=50, width=50, fx=50.0, fy=50.0, cx=25.0, cy=25.0, near=0.01, far=1.0
)

b.setup_renderer(intrinsics)
model_dir = os.path.join(b.utils.get_assets_dir(), "bop/ycbv/models")
meshes = []
for idx in range(1, 22):
    mesh_path = os.path.join(
        model_dir, "obj_" + "{}".format(idx).rjust(6, "0") + ".ply"
    )
    b.RENDERER.add_mesh_from_file(mesh_path, scaling_factor=1.0 / 1000.0)

b.RENDERER.add_mesh_from_file(
    os.path.join(b.utils.get_assets_dir(), "sample_objs/cube.obj"),
    scaling_factor=1.0 / 1000000000.0,
)


pose = b.t3d.transform_from_pos(jnp.array([-1.0, -1.0, 4.0]))
pose2 = pose @ b.distributions.gaussian_vmf_zero_mean(
    jax.random.PRNGKey(5), 0.05, 1000.0
)


b.show_pose("1", pose)
b.show_pose("2", pose2)

img1 = b.RENDERER.render(pose.reshape(-1, 4, 4), jnp.array([0]))[..., :3]
img2 = b.RENDERER.render(pose2.reshape(-1, 4, 4), jnp.array([0]))[..., :3]

b.clear()
b.show_cloud("c", img1.reshape(-1, 3))
b.show_cloud("d", img2.reshape(-1, 3), color=b.RED)

mask = (img1[:, :, 2] < intrinsics.far) * (img2[:, :, 2] < intrinsics.far)

transform = b.utils.find_least_squares_transform_between_clouds(
    img1[mask, :], img2[mask, :]
)

print(jnp.abs(img2[mask, :] - img1[mask, :]).sum())
print(jnp.abs(img2[mask, :] - b.apply_transform(img1[mask, :], transform)).sum())
print(jnp.abs(cloud_transformed - b.apply_transform(cloud, transform)).sum())
