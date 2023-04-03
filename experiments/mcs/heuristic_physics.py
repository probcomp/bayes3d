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
import glob
# j.meshcat.setup_visualizer()



# scene_name = "charlie_0002_04_B1_debug.json"


dx  = 0.7
dy = 0.7
dz = 0.7
gridding1 = j.make_translation_grid_enumeration(
    -dx, -dy, -dz, dx, dy, dz, 21,15,15
)

# dx  = 0.2
# dy = 0.2
# dz = 0.2
# gridding2 = j.make_translation_grid_enumeration(
#     -dx, -dy, -dz, dx, dy, dz, 7,7,7
# )
gridding = [gridding1]





R_SWEEP = jnp.array([0.03])
OUTLIER_PROB=0.05
OUTLIER_VOLUME=1.0



scene_name = "passive_physics_gravity_support_0001_26"

scene_name = "passive_physics_collision_0001_05"
scene_name = "passive_physics_collision_0001_03"
scene_name = "passive_physics_collision_0001_04"


scene_name = "passive_physics_spatio_temporal_continuity_0001_02"
scene_name = "passive_physics_spatio_temporal_continuity_0001_15"
scene_name = "passive_physics_spatio_temporal_continuity_0001_14"

scene_name = "passive_physics_spatio_temporal_continuity_0001_02"

scene_name = "passive_physics_shape_constancy_0001_06"

scene_name = "passive_physics_object_permanence_0001_29"
scene_name = "passive_physics_object_permanence_0001_01"
scene_name = "passive_physics_object_permanence_0001_02"
scene_name = "passive_physics_object_permanence_0001_03"

scene_regex = os.path.join(j.utils.get_assets_dir(), "mcs_scene_jsons", "eval_6_validation", "passive_physics_object_permanence_0001_28.json")
scene_regex = os.path.join(j.utils.get_assets_dir(), "mcs_scene_jsons", "eval_6_validation", "passive_physics_gravity_support*")

scene_regex = os.path.join(j.utils.get_assets_dir(), "mcs_scene_jsons", "eval_6_validation", "passive_physics_spatio_temporal_*.json")
scene_regex = os.path.join(j.utils.get_assets_dir(), "mcs_scene_jsons", "eval_6_validation", "passive_physics_object_permanence*")
scene_regex = os.path.join(j.utils.get_assets_dir(), "mcs_scene_jsons", "eval_6_validation", "passive_physics_shape_constancy_*")

scene_regex = os.path.join(j.utils.get_assets_dir(), "mcs_scene_jsons", "eval_6_validation", "passive_physics_gravity_support*")
scene_regex = os.path.join(j.utils.get_assets_dir(), "mcs_scene_jsons", "eval_6_validation", "passive_physics_collision_0001_05.json")

scene_regex = os.path.join(j.utils.get_assets_dir(), "mcs_scene_jsons", "eval_6_validation", "passive_physics_spatio_temporal_continuity*.json")

files = glob.glob(scene_regex)
files = [i.split("/")[-1] for i in files]

files = sorted(files)

physics_server = j.physics.PhysicsServer()

for scene_name in files:
    print(scene_name)
    scene_path = os.path.join(j.utils.get_assets_dir(), "mcs_scene_jsons", "eval_6_validation",
        scene_name
    )

    images = j.physics.load_mcs_scene_data(scene_path)
    images = images

    # j.make_gif([j.multi_panel([j.get_rgb_image(image.rgb)], [f"{i} / {len(images)}"]) for (i, image) in enumerate(images)], "rgb.gif")

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
    physics_server.reset(intrinsics)
    for image in images[1:]:
        physics_server.update(image)

    ALL_OBJECT_POSES = physics_server.ALL_OBJECT_POSES
    viz_images = []
    for t in range(len(images)):
        POSES = jnp.array(ALL_OBJECT_POSES[t])
        rerendered = physics_server.renderer.render_multiobject(POSES, jnp.arange(POSES.shape[0]))
        rerendered_viz = j.resize_image(j.get_depth_image(rerendered[:,:,3],max=ALL_OBJECT_POSES[-1].shape[0]+1), image.intrinsics.height, image.intrinsics.width)
        rgb_viz = j.get_rgb_image(images[t].rgb)
        viz = j.multi_panel(
                [
                    rgb_viz,
                    rerendered_viz,
                    j.overlay_image(rgb_viz, rerendered_viz)
                ],
            labels=["RGB", "Inferred", "Overlay"],
            title=f"{t} / {len(images)} - Num Objects : {POSES.shape[0]}/{ALL_OBJECT_POSES[-1].shape[0]} Plausibilty: {physics_server.plausibility[t]}"
        )
        viz_images.append(viz)
    j.make_gif(viz_images, f"{scene_name}.gif")
    print(scene_name)

    from IPython import embed; embed()



# def plausibility_calculation()

from IPython import embed; embed()


    # if len(known_objects) > 0:
    #     all_current_pose_estimates = jnp.array([k[-1][1] for k in known_objects]).reshape(-1,4,4)
    #     rerendered = renderer.render_multiobject(all_current_pose_estimates, jnp.arange(all_current_pose_estimates.shape[0]))
    #     rerendered_viz = j.resize_image(j.get_depth_image(rerendered[:,:,2],max=WALL_Z), image.intrinsics.height, image.intrinsics.width)
    #     rgb_viz = j.get_rgb_image(images[t].rgb)
    #     viz = j.multi_panel(
    #             [
    #                 rgb_viz,
    #                 rerendered_viz,
    #                 j.overlay_image(rgb_viz, rerendered_viz)
    #             ],
    #         labels=["RGB", "Inferred", "Overaly"],
    #         title=f"{t} / {len(images)}"
    #     )
    #     viz.save(f"{t}.png")
    #     viz.save(f"0.png") 