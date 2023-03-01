import machine_common_sense as mcs
import jax3dp3 as j
import numpy as np
import jax.numpy as jnp

controller = mcs.create_controller(
    "config_level2.ini"
)

scene_data = mcs.load_scene_json_file("charlie_0002_04_B1_debug.json")


step_metadata = controller.start_scene(scene_data)
image = j.RGBD.construct_from_step_metadata(step_metadata)


step_metadatas = [step_metadata]
for _ in range(200):
    step_metadata = controller.step("Pass")
    if step_metadata is None:
        break
    step_metadatas.append(step_metadata)

step_metadatas = [i for i in step_metadatas if i is not None]

images = [
     j.RGBD.construct_from_step_metadata(s) for s in step_metadatas
]
image = images[0]

j.setup_visualizer()
point_cloud_image = j.t3d.unproject_depth(image.depth, image.intrinsics)
objects = []
for i in jnp.unique(image.segmentation):
    objects.append(point_cloud_image[image.segmentation == i])

colors = j.distinct_colors(len(objects))
for i in range(len(objects)):
    j.show_cloud(f"{i}", objects[i],color=np.array(colors[i]))

import trimesh

meshes = []
for o in objects:
    meshes.append(trimesh.voxel.ops.points_to_marching_cubes(o, pitch=0.04))

renderer = j.Renderer(image.intrinsics)
for m in meshes:
    renderer.add_mesh(m)

initial_poses = jnp.array([jnp.eye(4) for _ in range(len(renderer.meshes))])


rerendered = renderer.render_multiobject(initial_poses, jnp.arange(len(renderer.meshes)))
depth_reconstruction = j.get_depth_image(rerendered[:,:,2], max=image.intrinsics.far)
depth_real = j.get_depth_image(image.depth, max=image.intrinsics.far)
j.vstack_images([depth_reconstruction, depth_real]).save("reconstruction.png")

d = 0.4
n = 10
translation_deltas_global = j.make_translation_grid_enumeration(
    -d, -d, -d, d, d, d, n, n, n
)

R_SWEEP = jnp.array([0.02])
OUTLIER_PROB=0.1
OUTLIER_VOLUME=1.0

pose_estimates = initial_poses
pose_estimates_all = [pose_estimates]

for t in range(len(step_metadatas))[:10]:
    print(t)
    image = j.RGBD.construct_from_step_metadata(step_metadatas[t])
    point_cloud_image = j.t3d.unproject_depth(image.depth, image.intrinsics)

    for i in jnp.arange(len(renderer.meshes)):
        pose_estimates_tiled = jnp.tile(pose_estimates[:,None,...], (1, translation_deltas_global.shape[0], 1, 1))
        pose_proposals = pose_estimates_tiled.at[i].set(jnp.einsum("aij,ajk->aik", pose_estimates_tiled[i,...], translation_deltas_global))
        rendered_images = renderer.render_multiobject_parallel(pose_proposals, jnp.arange(len(renderer.meshes)))[...,:3]
        weights = j.threedp3_likelihood_with_r_parallel_jit(
            point_cloud_image, rendered_images, R_SWEEP, OUTLIER_PROB, OUTLIER_VOLUME
        )
        pose_estimates = pose_proposals[:,weights[0,:].argmax(), :,:]

    pose_estimates_all.append(pose_estimates)

all_viz_images = []
for i in range(len(images)):
    image = images[i]
    rerendered = renderer.render_multiobject(pose_estimates_all[i], jnp.arange(len(renderer.meshes)))
    depth_reconstruction = j.get_depth_image(rerendered[:,:,2], max=image.intrinsics.far)
    depth_real = j.get_depth_image(image.depth, max=image.intrinsics.far)
    all_viz_images.append(j.vstack_images([depth_reconstruction, depth_real]))
j.make_gif(all_viz_images, "out.gif")

from IPython import embed; embed()


j.clear()
colors = j.distinct_colors(len(objects))
for i in range(len(renderer.meshes)):
    j.show_trimesh(f"{i}", renderer.meshes[i], color=np.array(colors[i]))


import time
for t in range(len(pose_estimates_all)):
    for i in range(len(objects)):
        j.set_pose(f"{i}", pose_estimates_all[t][i])
    time.sleep(0.1)