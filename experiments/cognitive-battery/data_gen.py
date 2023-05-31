import json
import os

import cog_utils as utils
import jax.numpy as jnp
import numpy as np
from PIL import Image

import bayes3d
from bayes3d.transforms_3d import transform_from_pos, unproject_depth
from bayes3d.viz import get_depth_image

scene = "gravity"
DATA_PREFIX = os.environ.get("JAX3DP3_DATA_PATH", "data/")
MESHES_PATH = os.path.join(DATA_PREFIX, "gravity_data/meshes")

data_path = f"{DATA_PREFIX}/gravity_data/videos/1_tubes/2/human_readable/"
num_frames = len(os.listdir(os.path.join(data_path, "frames")))

width = 300
height = 300
fov = 90

intrinsics = utils.get_camera_intrinsics(width, height, fov)
renderer = bayes3d.Renderer(intrinsics=intrinsics)

rgb_images, depth_images, seg_maps = [], [], []
rgb_images_pil = []
for i in range(num_frames):
    rgb_path = os.path.join(data_path, f"frames/frame_{i}.jpeg")
    if not os.path.isfile(rgb_path):
        rgb_path = rgb_path.replace("jpeg", "png")
    rgb_img = Image.open(rgb_path)
    rgb_images_pil.append(rgb_img)
    rgb_images.append(np.array(rgb_img))

    depth_path = os.path.join(data_path, f"depths/frame_{i}.npy")
    depth_npy = np.load(depth_path)
    depth_images.append(depth_npy)

    seg_map = np.load(os.path.join(data_path, f"segmented/frame_{i}.npy"))
    seg_maps.append(seg_map)

with open("resources/scene_crops.json") as f:
    crops = json.load(f)[scene]

coord_images = []  # depth data in 2d view as images
seg_images = []  # segmentation data as images

for frame_idx in range(num_frames):
    coord_image = np.array(unproject_depth(depth_images[frame_idx], intrinsics))
    segmentation_image = seg_maps[frame_idx].copy()
    mask = np.logical_and.reduce(
        [
            *[(coord_image[:, :, i] > crops[i][0]) for i in range(len(crops))],
            *[(coord_image[:, :, i] < crops[i][1]) for i in range(len(crops))],
        ]
    )
    mask = np.invert(mask)

    coord_image[mask, :] = 0.0
    segmentation_image[mask, :] = 0.0
    coord_images.append(coord_image)
    seg_images.append(segmentation_image)

coord_images = np.stack(coord_images)
seg_images = np.stack(seg_images)

# Load meshes
meshes = []
meshes_path = MESHES_PATH
for mesh_name in os.listdir(meshes_path):
    if not mesh_name.endswith(".obj"):
        continue
    mesh_path = os.path.join(meshes_path, mesh_name)
    renderer.add_mesh_from_file(mesh_path, force="mesh")
    meshes.append(mesh_name.replace(".obj", ""))

start_t = 36
seg_img = seg_images[start_t]

num_objects = 10
indices, init_poses = [], []
obj_ids = jnp.unique(seg_img.reshape(-1, 3), axis=0)
obj_ids = sorted(
    obj_ids, key=lambda x: jnp.all(seg_img == x, axis=-1).sum(), reverse=True
)
print("Count:", len(obj_ids))
for obj_id in obj_ids[: num_objects + 1]:
    if jnp.all(obj_id == 0):
        # Background
        continue

    obj_mask = jnp.all(seg_img == obj_id, axis=-1)
    masked_depth = coord_images[start_t].copy()
    masked_depth[~obj_mask] = 0

    object_points = coord_images[start_t][obj_mask]
    maxs = np.max(object_points, axis=0)
    mins = np.min(object_points, axis=0)
    dims = maxs - mins
    obj_center = (maxs + mins) / 2
    obj_transform = transform_from_pos(obj_center)

    best = utils.find_best_mesh(renderer, meshes, obj_transform, masked_depth)
    if best:
        indices.append(best[0])
        init_poses.append(best[1])

init_poses = jnp.array(init_poses)
rendered_image = renderer.render_multiobject(init_poses, indices)
get_depth_image(rendered_image[:, :, 2], max=5).save("rendered_datagen.png")
breakpoint()
np.savez(
    "data.npz",
    rgb_images=rgb_images[start_t:],
    depth_images=depth_images[start_t:],
    segmentation_images=seg_images[start_t:],
    camera_params=tuple(intrinsics),
    poses=init_poses,
)
