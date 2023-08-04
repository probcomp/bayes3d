import json
import os
from collections import deque

import bayes3d
import jax
import jax.numpy as jnp
import numpy as np
from bayes3d.renderer import _Renderer
from bayes3d.transforms_3d import transform_from_pos, unproject_depth
from bayes3d.viz import get_depth_image, make_gif_from_pil_images, multi_panel
from bayes3d.distributions import gaussian_vmf
from PIL import Image
from tqdm import tqdm

import cog_utils as utils
from config import Config


def model(
    config: Config,
    video_path: str,
    meshes_path: str,
    num_start_objects=10,
    out_path: str = "",
):
    data_path = os.path.join(video_path, "human_readable")
    if not os.path.isdir(data_path):
        data_path = video_path
    num_frames = len(os.listdir(os.path.join(data_path, "frames")))

    intrinsics = utils.get_camera_intrinsics(config.width, config.height, config.fov)
    bayes3d.setup_renderer(intrinsics)
    renderer: _Renderer = bayes3d.RENDERER

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
        crops = json.load(f)[config.scene]

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

        coord_image[mask, :] = intrinsics.far
        segmentation_image[mask, :] = 0.0
        coord_images.append(coord_image)
        seg_images.append(segmentation_image)

    coord_images = np.stack(coord_images)
    seg_images = np.stack(seg_images)

    # Load meshes
    meshes = []

    for mesh_name in os.listdir(meshes_path):
        if not mesh_name.endswith(".obj"):
            continue
        mesh_path = os.path.join(meshes_path, mesh_name)
        renderer.add_mesh_from_file(mesh_path, force="mesh")
        meshes.append(mesh_name.replace(".obj", ""))
    print("Found meshes:", meshes)

    start_t = config.start_t
    seg_img = seg_images[start_t]
    known_obj_ids = set()

    indices, init_poses = [], []
    obj_ids = np.unique(seg_img.reshape(-1, 3), axis=0)
    obj_ids = sorted(
        obj_ids, key=lambda x: np.all(seg_img == x, axis=-1).sum(), reverse=True
    )
    for obj_id in obj_ids[: num_start_objects + 1]:
        if np.all(obj_id == 0):
            # Background
            continue

        obj_mask = np.all(seg_img == obj_id, axis=-1)
        masked_depth = coord_images[start_t].copy()
        masked_depth[~obj_mask] = intrinsics.far

        object_points = coord_images[start_t][obj_mask]
        maxs = np.max(object_points, axis=0)
        mins = np.min(object_points, axis=0)
        obj_center = (maxs + mins) / 2
        obj_transform = transform_from_pos(obj_center)

        best = utils.find_best_mesh(
            renderer, intrinsics, meshes, obj_transform, masked_depth
        )
        if best:
            indices.append(best[0])
            init_poses.append(best[1])
            known_obj_ids.add(tuple(obj_id))

    init_poses = jnp.array(init_poses)
    num_objects = init_poses.shape[0]

    ## Defining inference helper functions

    # Enumerating proposals
    def make_unfiform_grid(n, d):
        # d: number of enumerated proposals on each dimension (x, y, z).
        # n: the minimum and maximum position delta on each dimension (x, y, z).
        return bayes3d.make_translation_grid_enumeration(-d, -d, -d, d, d, d, n, n, n)

    def prior(new_pose, prev_poses, ewma_alpha=0.15):
        new_pose = new_pose[:3, 3]
        new_pose = new_pose.at[1].set(jnp.minimum(new_pose[1], config.table_y))

        # for gravity, prior is to move downwards
        gravity_shift = jnp.array([0.0, config.gravity_shift_prior, 0.0])

        # for velocity, use exponentially-weighted moving average
        poses_diffs = jnp.diff(prev_poses[:, :3, 3], axis=0)
        velocity_shift = jnp.array(
            [
                (p * (ewma_alpha * (1 - ewma_alpha) ** i))
                for i, p in enumerate(poses_diffs[::-1])
            ]
        ).sum(axis=0)

        prev_pose = prev_poses[-1][:3, 3]
        prior_shifts = gravity_shift + velocity_shift
        weight = jax.scipy.stats.norm.logpdf(
            new_pose - (prev_pose + prior_shifts),
            loc=0,
            scale=jnp.array([0.1, 0.1, 0.01]),
        )
        return weight.sum()

    prior_parallel = jax.jit(jax.vmap(prior, in_axes=(0, None)))

    # Liklelihood model
    def scorer(rendered_image, gt, var=0.01, op=0.05, ov=1, fs=3):
        # Liklihood parameters
        # r: radius
        # op: outlier probability
        # ov: outlier volume
        weight = bayes3d.likelihood.threedp3_likelihood(
            gt, rendered_image, var, op, ov, fs
        )
        return weight

    scorer_parallel = jax.jit(jax.vmap(scorer, in_axes=(0, None)))
    # scorer_parallel = jax.vmap(scorer, in_axes=(0, None))

    num_steps = (
        (num_frames - start_t) if config.num_steps == "auto" else config.num_steps
    )
    reward_indices = utils.get_reward_indices(meshes, indices)

    containment_relations = {}
    inferred_poses = []
    pose_estimates = init_poses.copy()
    past_poses = {
        i: deque([pose_estimates[i]] * config.num_past_poses)
        for i in range(pose_estimates.shape[0])
    }
    for t in tqdm(range(start_t, start_t + num_steps), desc="Inference"):
        gt_image = jnp.array(coord_images[t])
        if not len(reward_indices):
            new_indices, new_poses = utils.detect_new_objects(
                renderer=renderer,
                seg_img=seg_images[t],
                coord_image=coord_images[t],
                meshes=meshes,
                known_obj_ids=known_obj_ids,
            )
            if new_indices:
                indices.extend(new_indices)
                pose_estimates = jnp.concatenate((pose_estimates, new_poses))
                past_poses.update(
                    {
                        (i + len(past_poses)): deque(
                            [new_poses[i]] * config.num_past_poses
                        )
                        for i in range(len(new_poses))
                    }
                )
                num_objects += len(new_indices)
                reward_indices = utils.get_reward_indices(meshes, indices)

        rotation_deltas = jax.vmap(lambda key: gaussian_vmf(key, 0.00001, 800.0))(
            jax.random.split(jax.random.PRNGKey(3), 100)
        )
        for _ in range(config.iterations_per_step):
            for i in range(num_objects):
                # Check for occlusion
                if i in reward_indices:
                    occluded = utils.check_occlusion(
                        renderer, pose_estimates, indices, i
                    )
                    if occluded:
                        containing_obj = utils.check_containment(
                            renderer, pose_estimates, indices, i
                        )
                        if containing_obj is not None:
                            containment_relations[containing_obj] = i

                for d in config.grid_deltas:
                    translation_deltas = make_unfiform_grid(n=config.grid_n, d=d)
                    for pose_deltas in [translation_deltas, rotation_deltas]:
                        pose_deltas_full = jnp.tile(
                            jnp.eye(4)[None, :, :],
                            (pose_deltas.shape[0], pose_estimates.shape[0], 1, 1),
                        )
                        pose_deltas_full = pose_deltas_full.at[:, i, :, :].set(
                            pose_deltas
                        )
                        pose_proposals = jnp.einsum(
                            "bij,abjk->abik", pose_estimates, pose_deltas_full
                        )
                        prior_score = prior_parallel(
                            pose_proposals[:, i], jnp.array(past_poses[i])
                        )
                        images = renderer.render_multiobject_parallel(
                            pose_proposals.transpose((1, 0, 2, 3)), indices
                        )

                        likelihood_score = scorer_parallel(images, gt_image)
                        weights = likelihood_score + prior_score
                        best_weight_idx = jnp.argmax(weights)
                        best_proposal = pose_proposals[best_weight_idx]
                        pose_estimates = best_proposal

                past_poses[i].append(pose_estimates[i])
                if len(past_poses[i]) > config.num_past_poses:
                    past_poses[i].popleft()

            for i, j in containment_relations.items():
                i_delta_pose = past_poses[i][-1] - past_poses[i][-2]
                new_pose_estimate = pose_estimates[j] + i_delta_pose
                pose_estimates = pose_estimates.at[j].set(new_pose_estimate)

        inferred_poses.append(pose_estimates.copy())

    reward_idx = reward_indices[-1]
    apple_pos = pose_estimates[reward_idx, :-1, 3]
    sorted_receptacles = sorted(
        [
            pose_estimates[i, :-1, 3]
            for i in range(len(indices))
            if meshes[indices[i]] == config.receptacle_name
        ],
        key=lambda x: x[0].item(),
    )
    closest_receptacle_idx = jnp.argmin(
        jnp.array(
            [
                abs(apple_pos[0] - receptacle_pos[0])
                for receptacle_pos in sorted_receptacles
            ]
        )
    )

    if out_path:
        all_images = []
        for t in range(start_t, start_t + num_steps):
            rgb_viz = Image.fromarray(rgb_images[t].astype(np.int8), mode="RGB")
            gt_depth_1 = get_depth_image(coord_images[t][:, :, 2], max=5.0)
            poses = inferred_poses[t - start_t]
            rendered_image = renderer.render_multiobject(poses, indices[: len(poses)])
            rendered_image_depth = get_depth_image(rendered_image[:, :, 2], max=5)
            rendered_image_segmentation = get_depth_image(
                rendered_image[:, :, -1], max=num_objects
            )

            all_images.append(
                multi_panel(
                    [
                        rgb_viz,
                        gt_depth_1,
                        rendered_image_depth,
                        rendered_image_segmentation,
                    ],
                    [
                        f"Class: {closest_receptacle_idx}\nRGB Image",
                        "\nActual Depth",
                        f"   Frame: {t}\nReconstructed Depth",
                        "\nReconstructed Segmentation",
                    ],
                    middle_width=10,
                    label_fontsize=20,
                )
            )
        make_gif_from_pil_images(all_images, out_path)
        print("Saved output to:", out_path)

    return closest_receptacle_idx
