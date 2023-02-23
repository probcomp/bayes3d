import logging
import os 
import random
import bpy
import kubric as kb
from kubric.renderer.blender import Blender as KubricRenderer

logging.basicConfig(level="INFO")

#create a sample scene from a list of object poses, position, orientation, and paths to obj files and render it 
def render_kb_scene(objs, scene_name: str, output_dir: str, resolution = (256, 256)):
    # --- create scene and attach a renderer to it
    scene = kb.Scene(resolution=resolution)
    renderer = KubricRenderer(scene)

    # --- populate the scene with floor, light, camera
    room_size = 9
    scene += kb.Cube(name="floor", scale=(room_size,room_size, 0.1), position=(0, 0, 0))
    scene += kb.DirectionalLight(name="sun", position=(0, 0, 5), look_at=(0, 0, 0), intensity=5)
    scene += kb.PerspectiveCamera(name="camera", position=(-8,-8, 3), look_at=(4, 4, 3))

    # --- create walls
    scene += kb.Cube(name="wall1", scale=(0.1, room_size, room_size/2), position=(room_size, 0, 0))
    scene += kb.Cube(name="wall2", scale=(0.1, room_size, room_size/2), position=(-room_size, 0, 0))
    scene += kb.Cube(name="wall3", scale=(room_size, 0.1, room_size/2), position=(0, room_size, 0))
    scene += kb.Cube(name="wall4", scale=(room_size, 0.1, room_size/2), position=(0, -room_size, 0))

    # --- set background from HDRI 
    hdri_source = kb.AssetSource.from_manifest("gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")
    train_backgrounds, test_backgrounds = hdri_source.get_test_split(fraction=0.1)
    logging.info("Choosing one of the %d training backgrounds...", len(train_backgrounds))
    restitution, friction = 0.5, 0.5
    kubasic = kb.AssetSource.from_manifest("gs://kubric-public/assets/KuBasic/KuBasic.json")
    hdri_id = random.choice(train_backgrounds)
    background_hdri = hdri_source.create(asset_id=hdri_id)
    logging.info("Using background %s", hdri_id)
    scene.metadata["background"] = hdri_id
    renderer._set_ambient_light_hdri(background_hdri.filename)

    # Dome
    dome = kubasic.create(asset_id="dome", name="dome",
                        friction=friction,
                        restitution=restitution,
                        static=True, background=True)
    assert isinstance(dome, kb.FileBasedObject)

    scene += dome
    dome_blender = dome.linked_objects[renderer]
    texture_node = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
    texture_node.image = bpy.data.images.load(background_hdri.filename)

    floor_material = kb.PrincipledBSDFMaterial(roughness=1., specular=0.)
    floor_material.color = kb.random_hue_color()
    scene.metadata["background"] = floor_material.color.hexstr

    # --- add objects to the scene
    for obj in objs:
        obj = kb.FileBasedObject(
            asset_id=obj["asset_id"], 
            render_filename=obj["render_filename"],
            bounds=((-1, -1, -1), (1, 1, 1)),
            simulation_filename=None,
            position=obj["position"],
            scale=obj["scale"],
            quaternion=obj["quaternion"],
        )
        scene += obj

    # --- render (and save the blender file)
    renderer.save_state(os.path.join(output_dir, scene_name + ".blend"))
    frame = renderer.render_still()
    kb.write_png(frame["rgba"], os.path.join(output_dir, scene_name + ".png"))
    kb.write_palette_png(frame["segmentation"], os.path.join(output_dir, scene_name + "_segmentation.png"))
    scale = kb.write_scaled_png(frame["depth"], os.path.join(output_dir, scene_name + "_depth.png"))
    logging.info("Depth scale: %s", scale)