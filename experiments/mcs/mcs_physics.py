import machine_common_sense as mcs
import jax3dp3 as j
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
import os
import jax

controller = mcs.create_controller(
    os.path.join(j.utils.get_assets_dir(), "mcs_scene_jsons",  "config_level2.ini")
)

scene_data = mcs.load_scene_json_file(os.path.join(j.utils.get_assets_dir(), "mcs_scene_jsons", "charlie_0002_04_B1_debug.json"))

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
images = images[94:]



image = images[0]



intrinsics = j.camera.scale_camera_parameters(image.intrinsics, 0.25)

renderer = j.Renderer(intrinsics)

point_cloud_image = j.t3d.unproject_depth(j.utils.resize(image.depth, intrinsics.height, intrinsics.width), intrinsics)
in_scene_mask = (point_cloud_image[:,:,2] < 14.45) * (point_cloud_image[:,:,1] < 1.46)
j.viz.get_depth_image(in_scene_mask, max=1.0).save("mask.png")

segmentation = (j.utils.resize(image.segmentation, intrinsics.height, intrinsics.width) + 1) * (in_scene_mask > 0)
objects = []
for i in jnp.unique(segmentation):
    if i == 0:
        continue
    objects.append(point_cloud_image[segmentation == i])
print(len(objects))

colors = j.distinct_colors(len(objects))
for i in range(len(objects)):
    j.meshcat.show_cloud(f"{i}", objects[i],color=np.array(colors[i]))





import trimesh

poses = []
meshes = []
for o in objects:
    dims, pose = j.utils.aabb(o)
    poses.append(pose)
    meshes.append(trimesh.voxel.ops.points_to_marching_cubes(j.t3d.apply_transform(o, j.t3d.inverse_pose(pose)), pitch=0.075))
NUM_OBJECTS = len(meshes)


j.meshcat.clear()
for i in range(len(meshes)):
    j.meshcat.show_trimesh(f"{i}", meshes[i])
    j.meshcat.set_pose(f"{i}", poses[i])





plane = j.mesh.make_cuboid_mesh([100.0, 100.0, 0.001])
wall_pose = j.t3d.transform_from_pos_target_up(
    jnp.array([0.0, 0.0, 14.5]),
    jnp.array([0.0, 0.0, 0.5]),
    jnp.array([0.0, -1.0, 0.0]),
)
floor_pose = j.t3d.transform_from_pos_target_up(
    jnp.array([0.0, 1.45, 0.0]),
    jnp.array([0.0, 0.0, 0.0]),
    jnp.array([0.0, 0.0, 1.0]),
)
renderer.add_mesh(plane)
for m in meshes:
    renderer.add_mesh(m)



initial_poses = jnp.vstack([jnp.array(poses),jnp.array([wall_pose, floor_pose])])


rerendered = renderer.render_multiobject(initial_poses, [*np.arange(NUM_OBJECTS)+1,0,0])
depth_reconstruction = j.get_depth_image(rerendered[:,:,2], max=image.intrinsics.far)
depth_real = j.get_depth_image(image.depth, max=image.intrinsics.far)
j.vstack_images([depth_reconstruction, depth_real]).save("reconstruction.png")



d = 0.3
n = 9
translation_deltas_global = j.make_translation_grid_enumeration(
    -d, -d, -d, d, d, d, n, n, n
)

key = jax.random.PRNGKey(3)

import jax
rotation_deltas_global = jax.vmap(
    lambda ang: j.t3d.transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), jnp.deg2rad(ang)))(jnp.linspace(-20.0, 20.0, 100))

R_SWEEP = jnp.array([0.1])
OUTLIER_PROB=0.01
OUTLIER_VOLUME=5.0

pose_estimates = initial_poses
pose_estimates_all = [pose_estimates]

for t in range(1,len(images)):
    image = images[t]
    print(t)
    point_cloud_image = j.t3d.unproject_depth(j.utils.resize(image.depth, intrinsics.height, intrinsics.width), intrinsics)
    for i in [*jnp.arange(NUM_OBJECTS), *jnp.arange(NUM_OBJECTS)]:
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
            rendered_images = renderer.render_multiobject_parallel(p, [*np.arange(NUM_OBJECTS)+1,0,0])[...,:3]
            weights = j.threedp3_likelihood_with_r_parallel_jit(
                point_cloud_image, rendered_images, R_SWEEP, OUTLIER_PROB, OUTLIER_VOLUME
            )
            all_weights.append(weights[0,:])
        all_pose_proposals = jnp.concatenate(all_pose_proposals, axis=1)
        all_weights = jnp.hstack(all_weights)

        pose_estimates = all_pose_proposals[:,all_weights.argmax(), :,:]


    pose_estimates_all.append(pose_estimates)

# assert len(images) == len(pose_estimates_all)

all_viz_images = []
for i in range(len(pose_estimates_all)):
    image = images[i]
    rerendered = renderer.render_multiobject(pose_estimates_all[i], [*np.arange(NUM_OBJECTS)+1,0,0])[...,:3]
    depth_reconstruction = j.get_depth_image(rerendered[:,:,2], max=image.intrinsics.far)
    depth_real = j.get_depth_image(image.depth, max=image.intrinsics.far)
    all_viz_images.append(j.vstack_images([depth_reconstruction, depth_real]))
j.make_gif(all_viz_images, "out.gif")
from IPython import embed; embed()


# j.meshcat.setup_visualizer()

# j.meshcat.clear()
# colors = j.distinct_colors(len(objects))
# for i in range(len(meshes)):
#     j.meshcat.show_trimesh(f"{i}",meshes[i],color=colors[i])

# t = 54
# point_cloud_image = jnp.array(j.t3d.unproject_depth(j.utils.resize(images[t].depth, intrinsics.height, intrinsics.width), intrinsics))
# j.meshcat.show_cloud("c", point_cloud_image.reshape(-1,3))
# rerendered = renderer.render_multiobject(pose_estimates_all[t], [*np.arange(NUM_OBJECTS)+1,0,0])[...,:3]
# j.meshcat.show_cloud("d", rerendered.reshape(-1,3), color=j.RED)

# for i in range(len(meshes)):
#     j.meshcat.set_pose(f"{i}",pose_estimates_all[t][i])
    

# import open3d as o3d
# def trimesh_to_o3d_triangle_mesh(trimesh_mesh):
#     mesh = o3d.geometry.TriangleMesh()
#     mesh.vertices =  o3d.utility.Vector3dVector(trimesh_mesh.vertices)
#     mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
#     return mesh



# viz = j.o3d_viz.O3DVis(intrinsics)

# viz.render.scene.clear_geometry()
# sphere = o3d.geometry.TriangleMesh.create_sphere(1.0)
# sphere.compute_vertex_normals()
# sphere.translate(np.array([0, 0, 13.5]))
# box = o3d.geometry.TriangleMesh.create_box(2, 4, 4)
# box.translate(np.array([-2.0, -1.0, 10.0]))
# box.compute_triangle_normals()

# mat_sphere = o3d.visualization.rendering.MaterialRecord()
# mat_sphere.shader = 'defaultLit'
# mat_sphere.base_color = [0.8, 0, 0, 1.0]

# mat_box = o3d.visualization.rendering.MaterialRecord()
# mat_box.shader = 'defaultLitTransparency'
# # mat_box.shader = 'defaultLitSSR'
# mat_box.base_color = [1.0, 0.2, 0.2, 0.4]
# mat_box.base_roughness = 0.0
# mat_box.base_reflectance = 0.0
# mat_box.base_clearcoat = 1.0
# mat_box.thickness = 1.0
# mat_box.transmission = 1.0
# mat_box.absorption_distance = 10
# mat_box.absorption_color = [0.5, 0.5, 0.5]


# viz.render.scene.remove_geometry(f"1")
# viz.render.scene.add_geometry(f"1", sphere, mat_sphere)

# viz.render.scene.remove_geometry(f"2")
# viz.render.scene.add_geometry(f"2", box, mat_box)

# j.get_rgb_image(viz.capture_image(image.intrinsics, np.eye(4))).save("test_open3d_viz.png")



# mat_box = o3d.visualization.rendering.MaterialRecord()  # or MaterialRecord(), for later versions of Open3D
# mat_box.shader = 'defaultLitTransparency'
# mat_box.base_color = [1.0, 0.467, 0.467, 0.2]
# mat_box.base_roughness = 0.0
# mat_box.base_reflectance = 0.0
# mat_box.base_clearcoat = 1.0
# mat_box.thickness = 1.0
# mat_box.transmission = 1.0
# mat_box.absorption_distance = 10
# mat_box.absorption_color = [0.5, 0.5, 0.5]

# viz.render.scene.clear_geometry()

# triangle_meshes_o3d = []
# for i in range(len(meshes)):
#     o3d_mesh = trimesh_to_o3d_triangle_mesh(meshes[i])
#     o3d_mesh.transform(pose_estimates_all[t][i])
#     o3d_mesh.compute_triangle_normals()

#     viz.render.scene.remove_geometry(f"{i}")
#     viz.render.scene.add_geometry(f"{i}", o3d_mesh, mat_box)


# light_dir = np.array([0.0, 0.0, 1.0])
# viz.render.scene.set_lighting(viz.render.scene.LightingProfile.NO_SHADOWS, (0, 0, 0))
# viz.render.scene.scene.remove_light('light2')
# viz.render.scene.scene.add_directional_light('light2',[1,1,1],light_dir,50000000.0,True)
# j.get_rgb_image(viz.capture_image(image.intrinsics, np.eye(4))).save("test_open3d_viz.png")






# import functools
# from functools import partial
# @functools.partial(
#     jnp.vectorize,
#     signature='(m)->()',
#     excluded=(1,2,3,4,),
# )
# def get_counts(
#     ij,
#     data_xyz: jnp.ndarray,
#     model_xyz: jnp.ndarray,
#     filter_size: int,
#     r,
# ):
#     dists = data_xyz[ij[0], ij[1], :3] - jax.lax.dynamic_slice(model_xyz, (ij[0], ij[1], 0), (2*filter_size + 1, 2*filter_size + 1, 3))
#     probs = (jnp.linalg.norm(dists, axis=-1) < r).sum()
#     return probs

# def threedp3_counts(
#     obs_xyz: jnp.ndarray,
#     rendered_xyz: jnp.ndarray,
#     r,
# ):
#     filter_size = 3
#     num_latent_points = obs_xyz.shape[1] * obs_xyz.shape[0]
#     rendered_xyz_padded = jax.lax.pad(rendered_xyz,  -100.0, ((filter_size,filter_size,0,),(filter_size,filter_size,0,),(0,0,0,)))
#     jj, ii = jnp.meshgrid(jnp.arange(obs_xyz.shape[1]), jnp.arange(obs_xyz.shape[0]))
#     indices = jnp.stack([ii,jj],axis=-1)
#     counts = get_counts(indices, obs_xyz, rendered_xyz_padded, filter_size, r)
#     return counts





# counts = threedp3_counts(point_cloud_image, rerendered, R_SWEEP[0]*4)
# j.get_depth_image(counts == 0).save("unexplained.png")





# idx = 2
# cloud = objects[idx]
# j.meshcat.clear()
# j.meshcat.show_cloud(f"1", objects[idx],color=np.array(colors[idx]))

# def is_non_object(entities):
#     distance_threshold = 0.07
#     pole_distance_threshold = 0.05
#     two_sides_threshold = 2.5
#     side_length_threshold = 1.7
#     min_side_length_threshold =  0.48


#     bounding_boxes = []
#     for e in entities:
#         bounding_boxes.append(
#             j.utils.aabb(e)
#         )

#     min_z = np.minimum([e[1][:,3][2] for e in bounding_boxes])

#     occluder = []
#     for (ent_id, (dims, p)) in enumerate(bounding_boxes):

#         if (p.pos[2] - min_z) < distance_threshold:
#             if (
#                 jnp.max(dims) > side_length_threshold or 
#                 jnp.min(dims) > min_side_length_threshold
#             ):
#                 occluder.append(ent_id)

#     bb_dist_threshold = 0.5
#     for (ent_id, (dims, p)) in enumerate(bounding_boxes):
#         for occluder_id in occluder
#             if ent_id in occluder:
#                 continue

#             if (p[:3,3][2] - min_z) < distance_threshold:
#                 corners = j.utils.bounding_box_corners(dims)
#                 corners = j.t3d.apply_transform(corners, p)
                

#     if (jnp.max(dims) > side_length_threshold or jnp.min(dims) > min_side_length_threshold):
#         return True

#     bb_dist_threshold = 0.5

#     if dims[1] < 0.1 or dims[1] > 0.1:
#         return True
    
#     pos = pose[:3,3]
#     if pos[2] > 14.4:
#         return True

    

    

#     return False


# from IPython import embed; embed()


# j.clear()
# colors = j.distinct_colors(len(objects))
# for i in range(len(renderer.meshes)):
#     j.show_trimesh(f"{i}", renderer.meshes[i], color=np.array(colors[i]))


# import time
# for t in range(len(pose_estimates_all)):
#     for i in range(len(objects)):
#         j.set_pose(f"{i}", pose_estimates_all[t][i])
#     time.sleep(0.1)