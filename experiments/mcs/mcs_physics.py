import machine_common_sense as mcs
import jax3dp3 as j
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
import os
import jax
import time
from tqdm import tqdm

# j.meshcat.setup_visualizer()



# scene_name = "charlie_0002_04_B1_debug.json"


scene_name = "passive_physics_spatio_temporal_continuity_0001_21"
scene_name = "passive_physics_gravity_support_0001_21"
scene_name = "passive_physics_shape_constancy_0001_06"
scene_name = "passive_physics_gravity_support_0001_24"
scene_name = "passive_physics_object_permanence_0001_41"
scene_name = "passive_physics_collision_0001_01"
if os.path.exists(f"{scene_name}.npz"):
    images = np.load(f"{scene_name}.npz",allow_pickle=True)["arr_0"]
else:
    controller = mcs.create_controller(
        os.path.join(j.utils.get_assets_dir(), "mcs_scene_jsons",  "config_level2.ini")
    )

    scene_data = mcs.load_scene_json_file(os.path.join(j.utils.get_assets_dir(), "mcs_scene_jsons", scene_name +".json"))

    step_metadata = controller.start_scene(scene_data)
    image = j.RGBD.construct_from_step_metadata(step_metadata)

    step_metadatas = [step_metadata]
    for _ in tqdm(range(200)):
        step_metadata = controller.step("Pass")
        if step_metadata is None:
            break
        step_metadatas.append(step_metadata)

    all_images = []
    for i in tqdm(range(len(step_metadatas))):
        all_images.append(j.RGBD.construct_from_step_metadata(step_metadatas[i]))

    images = all_images
    np.savez(f"{scene_name}.npz", images) 

# images = images[94:]

image = images[0]

rgbs = []
for i in range(len(images)):
    rgbs.append(j.multi_panel([j.get_rgb_image(images[i].rgb)], [f"{i} / {len(images)}"])) 
j.make_gif(rgbs, "rgb.gif")


intrinsics = j.camera.scale_camera_parameters(image.intrinsics, 0.25)

WALL_Z = 14.5
FLOOR_Y = 1.45


point_cloud_image = j.t3d.unproject_depth(j.utils.resize(image.depth, intrinsics.height, intrinsics.width), intrinsics)
in_scene_mask = (point_cloud_image[:,:,2] < WALL_Z-0.05) * (point_cloud_image[:,:,1] < FLOOR_Y - 0.05)
j.viz.get_depth_image(in_scene_mask, max=1.0).save("mask.png")

# segmentation = j.utils.segment_point_cloud_image(point_cloud_image * in_scene_mask[...,None], threshold=0.1)
# j.viz.get_depth_image(segmentation+1, max=segmentation.max()+1).save("seg.png")

segmentation = (j.utils.resize(image.segmentation, intrinsics.height, intrinsics.width) + 1) * (in_scene_mask > 0) - 1
objects = []
for i in jnp.unique(segmentation):
    if i == -1:
        continue
    objects.append(point_cloud_image[segmentation == i])
print(len(objects))

initial_object_poses = []
meshes = []
for o in objects:
    dims, pose = j.utils.aabb(o)
    initial_object_poses.append(pose)
    mesh = j.mesh.make_marching_cubes_mesh_from_point_cloud(
        j.t3d.apply_transform(o, j.t3d.inverse_pose(pose)),
        0.075
    )
    meshes.append(mesh)

plane = j.mesh.make_cuboid_mesh([100.0, 100.0, 0.001])
wall_pose = j.t3d.transform_from_pos_target_up(
    jnp.array([0.0, 0.0, WALL_Z]),
    jnp.array([0.0, 0.0, 0.5]),
    jnp.array([0.0, -1.0, 0.0]),
)
floor_pose = j.t3d.transform_from_pos_target_up(
    jnp.array([0.0, FLOOR_Y, 0.0]),
    jnp.array([0.0, 0.0, 0.0]),
    jnp.array([0.0, 0.0, 1.0]),
)
renderer = j.Renderer(intrinsics)
renderer.add_mesh(plane)
for m in meshes:
    renderer.add_mesh(m)



# j.meshcat.setup_visualizer()

if len(initial_object_poses)>0:
    initial_poses = jnp.vstack([jnp.array([wall_pose, floor_pose]),jnp.array(initial_object_poses)])
else:
    initial_poses = jnp.array([wall_pose, floor_pose])

def render_func(pose_estimates):
    return renderer.render_multiobject(pose_estimates, [0,0, *np.arange(len(pose_estimates)-2)+1,])

def render_func_parallel(pose_estimates):
    return renderer.render_multiobject_parallel(pose_estimates, [0,0, *np.arange(len(pose_estimates)-2)+1,])

rerendered = render_func(initial_poses)
# j.meshcat.show_cloud("1", rerendered.reshape(-1,3))
depth_reconstruction = j.get_depth_image(rerendered[:,:,2], max=intrinsics.far)
depth_real = j.get_depth_image(image.depth, max=image.intrinsics.far)
j.vstack_images([depth_reconstruction, depth_real]).save("reconstruction.png")


d = 0.3
n = 7
translation_deltas_global = j.make_translation_grid_enumeration(
    -d, -d, -d, d, d, d, n, n, n
)

key = jax.random.PRNGKey(3)

import jax
rotation_deltas_global1 = jax.vmap(
    lambda ang: j.t3d.transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.deg2rad(ang)))(jnp.linspace(-20.0, 20.0, 50))

rotation_deltas_global2 = jax.vmap(
    lambda ang: j.t3d.transform_from_axis_angle(jnp.array([0.0, 1.0, 0.0]), jnp.deg2rad(ang)))(jnp.linspace(-20.0, 20.0, 50))

rotation_deltas_global = jnp.vstack([rotation_deltas_global1,rotation_deltas_global2])


R_SWEEP = jnp.array([0.1])
OUTLIER_PROB=0.01
OUTLIER_VOLUME=5.0



def prior(new_pose, prev_pose):
    weight = jax.scipy.stats.norm.pdf(new_pose[:3,3] - (prev_pose[:3,3] + jnp.array([0.0, 0.2, 0.0])), loc=0, scale=0.1).sum()
    weight *= (new_pose[:3,3][1] < 1.4)
    return weight

prior_parallel = jax.jit(jax.vmap(prior, in_axes=(0, None)))

pose_estimates = initial_poses
pose_estimates_all = [pose_estimates]

for t in range(1,len(images)):
    image = images[t]
    print(t)
    point_cloud_image = j.t3d.unproject_depth(j.utils.resize(image.depth, intrinsics.height, intrinsics.width), intrinsics)

    for i in [*jnp.arange(pose_estimates.shape[0]), *jnp.arange(pose_estimates.shape[0])]:
        if i == 0 or i==1:
            continue
        pose_estimates_tiled = jnp.tile(
            pose_estimates[:,None,...], (1, translation_deltas_global.shape[0], 1, 1)
        )
        pose_proposals_translation = pose_estimates_tiled.at[i].set(
            jnp.einsum("aij,ajk->aik", pose_estimates_tiled[i,...], translation_deltas_global)
        )
        
        pose_estimates_tiled = jnp.tile(pose_estimates[:,None,...], (1, rotation_deltas_global.shape[0], 1, 1))
        pose_proposals_rotations = pose_estimates_tiled.at[i].set(jnp.einsum("aij,ajk->aik", pose_estimates_tiled[i,...], rotation_deltas_global))

        all_pose_proposals = [pose_proposals_translation, pose_proposals_rotations]
        all_weights = []
        for p in all_pose_proposals:
            rendered_images = render_func_parallel(p)[...,:3]
            weights = j.threedp3_likelihood_with_r_parallel_jit(
                point_cloud_image, rendered_images, R_SWEEP, OUTLIER_PROB, OUTLIER_VOLUME
            )
            all_weights.append(weights[0,:])
        all_pose_proposals = jnp.concatenate(all_pose_proposals, axis=1)
        
        all_weights = jnp.hstack(all_weights)
        all_weights += prior_parallel(all_pose_proposals[i], pose_estimates[i])
        

        pose_estimates = all_pose_proposals[:,all_weights.argmax(), :,:]

    image = images[t]
    point_cloud_image = j.t3d.unproject_depth(j.utils.resize(image.depth, intrinsics.height, intrinsics.width), intrinsics)
    j.get_depth_image(point_cloud_image[:,:,2],max=WALL_Z).save("img.png")
    rerendered = render_func(pose_estimates)[...,:3]
    j.get_depth_image(rerendered[:,:,2],max=WALL_Z).save("rerendered.png")
    counts = j.threedp3_counts(point_cloud_image, rerendered, R_SWEEP[0]*4)
    j.get_depth_image(counts,max=counts.max()).save("counts.png")
    unexplained = (counts == 0) * (point_cloud_image[:,:,2] < WALL_Z-0.05) * (point_cloud_image[:,:,1] < FLOOR_Y - 0.05)
    j.get_depth_image(unexplained).save("unexplained.png")

    seg = j.utils.segment_point_cloud_image(point_cloud_image * unexplained[...,None], threshold=0.3, min_points_in_cluster=5)
    unique_segs = jnp.unique(seg)
    unique_segs = unique_segs[unique_segs != -1]

    if len(unique_segs) > 0:
        
        BUFFER = 6
        poses = jnp.zeros((0,4,4))
        new_meshes = []
        for id in unique_segs:
            rows, cols = jnp.where(seg == id)
            if (rows < 0 + BUFFER).any() or (rows > intrinsics.height - BUFFER).any():
                continue

            if (cols < 0 + BUFFER).any() or (cols > intrinsics.width - BUFFER).any():
                continue

            o = point_cloud_image[seg == id]
            dims, pose = j.utils.aabb(o)
            poses = jnp.vstack(
                [
                    poses,
                    jnp.array([pose])
                ]
            )
            new_mesh = j.mesh.make_marching_cubes_mesh_from_point_cloud(j.t3d.apply_transform(o, j.t3d.inverse_pose(pose)), 0.075)
            new_meshes.append(new_mesh)
            print("Adding mesh!")

        print("before")
        for new_mesh in new_meshes:
            meshes.append(new_mesh)
        
        np.savez("meshes.npz", meshes)

        for new_mesh in new_meshes:
            renderer.add_mesh(new_mesh)
        print("after")

        pose_estimates = jnp.concatenate([pose_estimates, jnp.array(poses)])

    pose_estimates_all.append(pose_estimates)




# renderer = j.Renderer(intrinsics)
# renderer.add_mesh(plane)
# for m in meshes:
#     renderer.add_mesh(m)


viz_1 = []
for t in tqdm(range(len(pose_estimates_all))):
    image = images[t]
    point_cloud_image = j.t3d.unproject_depth(j.utils.resize(image.depth, intrinsics.height, intrinsics.width), intrinsics)
    rerendered = render_func(pose_estimates_all[t])[...,:3]
    counts = j.threedp3_counts(point_cloud_image, rerendered, R_SWEEP[0]*4)
    unexplained = (counts == 0) * (point_cloud_image[:,:,2] < WALL_Z-0.05) * (point_cloud_image[:,:,1] < FLOOR_Y - 0.05)
    viz_1.append(
        [
            j.get_rgb_image(image.rgb),
            j.resize_image(
                j.get_depth_image(point_cloud_image[:,:,2], max=WALL_Z),
                image.intrinsics.height,
                image.intrinsics.width
            ),
            j.resize_image(
                j.get_depth_image(unexplained),
                image.intrinsics.height,
                image.intrinsics.width
            )
        ]
    )

j.make_gif([j.hstack_images(i) for i in viz_1],"out.gif")


viz = j.o3d_viz.O3DVis(image.intrinsics)

# viz.render.scene.scene.remove_light("light")
# viz.render.scene.scene.add_directional_light('light',[1,1,1],np.array([0.0, 0.0, 1.0]),50000.0,True)
viz.render.scene.set_lighting(viz.render.scene.LightingProfile.NO_SHADOWS, (0, 0, 0))

colors = np.array(j.distinct_colors(len(meshes), pastel_factor=0.1))
viz_2 = []
for t in tqdm(range(len(pose_estimates_all))):
    # t = 105
    viz.clear()
    pose_estimates = pose_estimates_all[t]
    for i in range(2,len(pose_estimates)):
        viz.make_trimesh(meshes[i-2], np.array(pose_estimates[i]), [*colors[i-2], 0.8])


    viz.set_background(np.array([1.0, 1.0, 1.0, 1.0]))
    rgb = viz.capture_image(image.intrinsics, np.eye(4))
    
    
    viz_2.append(
        [
            j.get_rgb_image(rgb)
        ]
    )


viz_final = [None for _  in range(len(pose_estimates_all))]
for t in tqdm(range(len(pose_estimates_all))):
    viz_final[t] = j.multi_panel(
        [viz_1[t][0],viz_1[t][1],viz_2[t][0],viz_1[t][2]],
        labels=["RGB", "Depth", "Inferred", "Unexplained"]
    )

j.make_gif(viz_final, f"{scene_name}.gif")





from IPython import embed; embed()

# Visualizations




