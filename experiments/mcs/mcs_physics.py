import machine_common_sense as mcs
import jax3dp3 as j
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm

controller = mcs.create_controller(
    "config_level2.ini"
)

scene_data = mcs.load_scene_json_file("charlie_0002_04_B1_debug.json")

step_metadata = controller.start_scene(scene_data)
image = j.RGBD.construct_from_step_metadata(step_metadata)

step_metadatas = [step_metadata]
for _ in tqdm(range(200)):
    step_metadata = controller.step("Pass")
    if step_metadata is None:
        break
    step_metadatas.append(step_metadata)

images = []
for i in tqdm(range(len(step_metadatas))):
    images.append(j.RGBD.construct_from_step_metadata(step_metadatas[i]))


j.meshcat.setup_visualizer()

image = images[1]
point_cloud_image = j.t3d.unproject_depth(image.depth, image.intrinsics)
objects = []
for i in jnp.unique(image.segmentation):
    objects.append(point_cloud_image[image.segmentation == i])

colors = j.distinct_colors(len(objects))
for i in range(len(objects)):
    j.meshcat.show_cloud(f"{i}", objects[i],color=np.array(colors[i]))

idx = 2
cloud = objects[idx]
j.meshcat.clear()
j.meshcat.show_cloud(f"1", objects[idx],color=np.array(colors[idx]))

def is_non_object(entities):
    distance_threshold = 0.07
    pole_distance_threshold = 0.05
    two_sides_threshold = 2.5
    side_length_threshold = 1.7
    min_side_length_threshold =  0.48


    bounding_boxes = []
    for e in entities:
        bounding_boxes.append(
            j.utils.aabb(e)
        )

    min_z = np.minimum([e[1][:,3][2] for e in bounding_boxes])

    occluder = []
    for (ent_id, (dims, p)) in enumerate(bounding_boxes):

        if (p.pos[2] - min_z) < distance_threshold:
            if (
                jnp.max(dims) > side_length_threshold or 
                jnp.min(dims) > min_side_length_threshold
            ):
                occluder.append(ent_id)

    bb_dist_threshold = 0.5
    for (ent_id, (dims, p)) in enumerate(bounding_boxes):
        for occluder_id in occluder
            if ent_id in occluder:
                continue

            if (p[:3,3][2] - min_z) < distance_threshold:
                corners = j.utils.bounding_box_corners(dims)
                corners = j.t3d.apply_transform(corners, p)
                

    if (jnp.max(dims) > side_length_threshold or jnp.min(dims) > min_side_length_threshold):
        return True

    bb_dist_threshold = 0.5

    if dims[1] < 0.1 or dims[1] > 0.1:
        return True
    
    pos = pose[:3,3]
    if pos[2] > 14.4:
        return True

    

    

    return False


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