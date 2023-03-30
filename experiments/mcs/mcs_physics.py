import jax3dp3 as j
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
import os
import jax
import time
from tqdm import tqdm
import jax
import matplotlib.pyplot as plt

# j.meshcat.setup_visualizer()



# scene_name = "charlie_0002_04_B1_debug.json"


scene_name = "passive_physics_gravity_support_0001_26"

scene_name = "passive_physics_spatio_temporal_continuity_0001_07"
scene_name = "passive_physics_object_permanence_0001_02"
scene_name = "passive_physics_shape_constancy_0001_02"
scene_path = os.path.join(j.utils.get_assets_dir(), "mcs_scene_jsons", "eval_6_validation",
  scene_name + ".json"
)

images = j.physics.load_mcs_scene_data(scene_path)
images = images

j.make_gif([j.multi_panel([j.get_rgb_image(image.rgb)], [f"{i} / {len(images)}"]) for (i, image) in enumerate(images)], "rgb.gif")

WALL_Z = 14.5
FLOOR_Y = 1.45

image = images[0]
intrinsics = j.camera.scale_camera_parameters(image.intrinsics, 0.25)
intrinsics = j.Intrinsics(
    intrinsics.height, intrinsics.width,
    intrinsics.fx,
    intrinsics.fy,
    intrinsics.cx,
    intrinsics.cy,
    intrinsics.near,
    WALL_Z
)

def get_object_mask(point_cloud_image, segmentation, segmentation_ids):
    object_mask = jnp.zeros(point_cloud_image.shape[:2])
    object_ids = []
    for id in segmentation_ids:
        point_cloud_segment = point_cloud_image[segmentation == id]
        bbox_dims, pose = j.utils.aabb(point_cloud_segment)
        is_occluder = jnp.logical_or(jnp.logical_or(jnp.logical_or(jnp.logical_or(
            (bbox_dims[0] < 0.1),
            (bbox_dims[1] < 0.1)),
            (bbox_dims[1] > 1.1)),
            (bbox_dims[0] > 1.1)),
            (bbox_dims[2] > 2.1)
        )
        if not is_occluder:
            object_mask += (segmentation == id)
            object_ids.append(id)

    object_mask = jnp.array(object_mask) > 0
    return object_ids, object_mask


def get_new_shape_model(point_cloud_image, segmentation, seg_id):
    num_pixels = jnp.sum(segmentation == seg_id)
    rows, cols = jnp.where(segmentation == seg_id)
    distance_to_edge_1 = min(jnp.abs(rows - 0).min(), jnp.abs(rows - intrinsics.height).min())
    distance_to_edge_2 = min(jnp.abs(cols - 0).min(), jnp.abs(cols - intrinsics.width).min())

    point_cloud_segment = point_cloud_image[segmentation == seg_id]
    dims, pose = j.utils.aabb(point_cloud_segment)

    BUFFER = 1
    if num_pixels < 14:
        return None
    if distance_to_edge_1 < BUFFER or distance_to_edge_2 < BUFFER:
        return None

    resolution = 0.01
    voxelized = jnp.rint(point_cloud_segment / resolution).astype(jnp.int32)
    min_z = voxelized[:,2].min()
    depth = voxelized[:,2].max() - voxelized[:,2].min()

    front_face = voxelized[voxelized[:,2] <= min_z+20, :]

    slices = []
    for i in range(depth):
        slices.append(front_face + jnp.array([0.0, 0.0, i]))
    full_shape = jnp.vstack(slices) * resolution


    dims, pose = j.utils.aabb(full_shape)
    mesh = j.mesh.make_marching_cubes_mesh_from_point_cloud(
        j.t3d.apply_transform(full_shape, j.t3d.inverse_pose(pose)),
        0.075
    )

    # j.meshcat.setup_visualizer()
    # j.meshcat.show_cloud("1", point_cloud_segment)
    # j.meshcat.show_cloud("1", full_shape)

    print("Adding mesh!  Seg ID: ", seg_id, "Pixels: ",num_pixels, " dists: ", distance_to_edge_1, " ", distance_to_edge_2, " Pose: ", pose[:3, 3])

    return mesh, pose

R_SWEEP = jnp.array([0.1])
OUTLIER_PROB=0.1
OUTLIER_VOLUME=1.0


covariance = jnp.diag(jnp.array([0.01, 0.01, 0.01]))
def proposal(key, particle, particle_vel, bbox_dims):
    noisy_particle_vel = jax.random.multivariate_normal(
        key, particle_vel, covariance
    )
    rot = j.distributions.vmf(key, 500.0)
    new_particle = j.t3d.transform_from_pos(noisy_particle_vel) @ particle @ rot
    return new_particle, noisy_particle_vel

def prior(new_particle_vel, particle, particle_vel, bbox_dims):
    logscore = jax.scipy.stats.multivariate_normal.logpdf(
        new_particle_vel, particle_vel, covariance
    )
    return logscore

proposal_parallel = jax.jit(jax.vmap(proposal, in_axes=(0, 0, 0, None)))
prior_parallel = jax.jit(jax.vmap(prior, in_axes=(0, 0, 0, None)))


NUM_PARTICLES = 1000
keys = jax.random.split(jax.random.PRNGKey(3), NUM_PARTICLES)

renderer = j.Renderer(intrinsics)
PARTICLES = jnp.zeros((1, 0, NUM_PARTICLES, 4, 4))
PARTICLES_VEL = jnp.zeros((1, 0, NUM_PARTICLES, 3))
for t in range(1,len(images)):
    image = images[t]
    print("Time: ", t, "  -  ")

    depth = j.utils.resize(image.depth, intrinsics.height, intrinsics.width)
    point_cloud_image = j.t3d.unproject_depth(depth, intrinsics)
    
    segmentation = j.utils.resize(image.segmentation, intrinsics.height, intrinsics.width)
    segmentation_ids = jnp.unique(segmentation)

    object_ids, object_mask = get_object_mask(point_cloud_image, segmentation, segmentation_ids)
    
    depth_complement = depth * (1.0 - object_mask) + intrinsics.far * (object_mask)
    point_cloud_image_complement = j.t3d.unproject_depth(depth_complement, intrinsics)

    if PARTICLES.shape[1] > 0:
        obj_id = 0

        NEW_PARTICLES, NEW_VELS = proposal_parallel(keys, PARTICLES[-1, obj_id], PARTICLES_VEL[-1, obj_id], renderer.model_box_dims[obj_id])
        keys = jax.random.split(keys[0], NUM_PARTICLES)

        rendered_images = renderer.render_parallel(NEW_PARTICLES, obj_id)[...,:3]
        rendered_images_spliced = j.splice_image_parallel(rendered_images, point_cloud_image_complement)
        weights = j.threedp3_likelihood_with_r_parallel_jit(
            point_cloud_image, rendered_images_spliced, R_SWEEP, OUTLIER_PROB, OUTLIER_VOLUME
        ).reshape(-1)

        prior_weights = prior(NEW_VELS, PARTICLES[-1, obj_id], PARTICLES_VEL[-1, obj_id], renderer.model_box_dims[obj_id])
        weights += prior_weights

        PARTICLES = jnp.concatenate([
            PARTICLES,
            NEW_PARTICLES[None, None, ...]
        ], axis=0)

        PARTICLES_VEL = jnp.concatenate([
            PARTICLES_VEL,
            NEW_VELS[None, None, ...]
        ], axis=0)

        parent_idxs = jax.random.categorical(keys[0], weights, shape=weights.shape)
        keys = jax.random.split(keys[0], weights.shape[0])
        PARTICLES = PARTICLES[:,:,parent_idxs]
        PARTICLES_VEL = PARTICLES_VEL[:,:,parent_idxs]
    else:
        PARTICLES = jnp.vstack([
            PARTICLES,
            PARTICLES[-1][None, ...]
        ])
        PARTICLES_VEL = jnp.vstack([
            PARTICLES_VEL,
            PARTICLES_VEL[-1][None, ...]
        ])

    for seg_id in object_ids:
        if len(object_ids) > PARTICLES.shape[1]:
            data = get_new_shape_model(point_cloud_image, segmentation, seg_id)
            if data is None:
                continue
            mesh, pose = data
            renderer.add_mesh(mesh)
            particles_for_new_object = jnp.tile(
                pose[None, None, ...],
                (PARTICLES.shape[0], 1, NUM_PARTICLES, 1,1)
            )

            PARTICLES = jnp.concatenate([PARTICLES, particles_for_new_object], axis=1)
            PARTICLES_VEL = jnp.concatenate([PARTICLES_VEL, 
                jnp.zeros((PARTICLES_VEL.shape[0], 1, NUM_PARTICLES, 3))
            ],
                axis=1
            )

    print(PARTICLES.shape)
    print(PARTICLES_VEL.shape)

viz_images = []
max_objects = PARTICLES[-1].shape[0]
for t in tqdm(range(len(PARTICLES))):
    pose_estimates = PARTICLES[t][:,0,...]
    seg_rendered = renderer.render_multiobject(pose_estimates, np.arange(pose_estimates.shape[0]))[...,3]

    inferred = j.resize_image(
        j.get_depth_image(seg_rendered,max=max_objects+1),
        images[t].intrinsics.height,
        images[t].intrinsics.width
    )
    rgb = j.get_rgb_image(images[t].rgb)
    viz_images.append(j.multi_panel(
        [
            rgb,
            inferred,
            j.overlay_image(rgb, inferred)
        ],
        labels=["RGB", "Inferred"],
        title=f"{t} / {len(images)}"
    ))
j.make_gif(viz_images, f"{scene_name}.gif")
print(scene_name)



from IPython import embed; embed()


j.meshcat.setup_visualizer()


colors = np.array(j.distinct_colors(len(PARTICLES[-1]), pastel_factor=0.3))

j.meshcat.clear()

for (i,m) in enumerate(renderer.meshes):
    j.meshcat.show_trimesh(f"{i}", m, color=colors[i])

t = 110
image = images[t]
depth = j.utils.resize(image.depth, intrinsics.height, intrinsics.width)
point_cloud_image = j.t3d.unproject_depth(depth, intrinsics)
j.meshcat.show_cloud("cloud1", point_cloud_image.reshape(-1,3))

pose_estimates = PARTICLES[t][:,0,...]
cloud = renderer.render_multiobject(pose_estimates, np.arange(pose_estimates.shape[0]))[...,:3]
j.meshcat.show_cloud("cloud2", cloud.reshape(-1,3), color=j.RED)


for t in range(len(images)):
    print("Time: ", t, "  -  ")
    image = images[t]
    point_cloud_image = j.t3d.unproject_depth(depth, intrinsics)
    
for t in tqdm(range(len(PARTICLES_OVER_TIME))):
    for i in range(len(PARTICLES_OVER_TIME[t])):
        j.meshcat.set_pose(f"{i}", PARTICLES_OVER_TIME[t][i, 0])
    time.sleep(0.05)
