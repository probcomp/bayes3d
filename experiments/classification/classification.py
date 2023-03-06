
import numpy as np
import os
import jax
import jax.numpy as jnp
import trimesh
import time
import pickle
import jax3dp3.transforms_3d as t3d
import jax3dp3 as j
from dataclasses import dataclass
import sys
import warnings
import pybullet_planning
import cv2
import collections
import heapq

sys.path.extend(["/home/nishadgothoskar/ptamp/pybullet_planning"])
sys.path.extend(["/home/nishadgothoskar/ptamp"])
warnings.filterwarnings("ignore")

test_pkl_file = os.path.join(j.utils.get_assets_dir(),"sample_imgs/red_lego_multi.pkl")
test_pkl_file = os.path.join(j.utils.get_assets_dir(),"sample_imgs/lego_learning.pkl")
test_pkl_file = os.path.join(j.utils.get_assets_dir(),"sample_imgs/spoon_learning.pkl")
test_pkl_file = os.path.join(j.utils.get_assets_dir(),"sample_imgs/knife_spoon_box_real.pkl")
test_pkl_file = os.path.join(j.utils.get_assets_dir(),"sample_imgs/knife_spoon_real.pkl")
test_pkl_file = os.path.join(j.utils.get_assets_dir(),"sample_imgs/knife_spoon_box_real.pkl")
test_pkl_file = os.path.join(j.utils.get_assets_dir(),"sample_imgs/knife_sim.pkl")

test_pkl_file = os.path.join(j.utils.get_assets_dir(),"sample_imgs/demo2_nolight.pkl")

test_pkl_file = os.path.join(j.utils.get_assets_dir(),"sample_imgs/strawberry_error.pkl")

file = open(test_pkl_file,'rb')
camera_images = pickle.load(file)["camera_images"]

images = [j.RGBD.construct_from_camera_image(c) for c in camera_images]
intrinsics = j.camera.scale_camera_parameters(images[0].intrinsics, 0.3)
renderer = j.Renderer(intrinsics)

top_level_dir = os.path.dirname(os.path.dirname(pybullet_planning.__file__))
mesh_names = ["knife", "spoon", "cracker_box", "strawberry", "mustard_bottle", "sugar_box","banana"]
model_paths = [
    os.path.join(top_level_dir,"models/srl/ycb/032_knife/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/031_spoon/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/003_cracker_box/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/012_strawberry/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/006_mustard_bottle/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/004_sugar_box/textured.obj"),
    os.path.join(top_level_dir,"models/srl/ycb/011_banana/textured.obj"),
]
for (path, n) in zip(model_paths, mesh_names):
    renderer.add_mesh(j.mesh.center_mesh(trimesh.load(path)),n)

image = images[0]
all_hypotheses, obs_point_cloud_image, table_pose  = j.run_classification(image, renderer)



# j.run_occlusion_search(image, renderer, 3)






from IPython import embed; embed()

i = 0
scores = jnp.array([i[0] for i in all_hypotheses[i][-1]])
order = np.argsort(-np.array(scores))
(score, obj_idx, _, pose) = all_hypotheses[i][-1][order[0]]

dims = j.utils.aabb(renderer.meshes[obj_idx].vertices)[0]


np.save(os.path.join(j.utils.get_assets_dir(), "viz_debugging.npy"), (dims, pose, image.intrinsics))


cube_mesh = j.mesh.make_cuboid_mesh(dims)
renderer.add_mesh(cube_mesh)

img = renderer.render_single_object(pose, len(renderer.meshes) - 1)
img_viz = j.resize_image(j.get_depth_image(img[:,:,2]), image.intrinsics.height, image.intrinsics.width)
rgb_viz = j.get_rgb_image(image.rgb)
j.overlay_image(img_viz, rgb_viz).save("rgb.png")



j.o3d_viz.setup(images[0].intrinsics)

j.o3d_viz.clear()
j.o3d_viz.make_bounding_box(j.utils.aabb(renderer.meshes[obj_idx].vertices)[0], pose, None)
j.o3d_viz.set_camera(image.intrinsics, jnp.eye(4))
bbox_overlay = j.o3d_viz.capture_image()
alpha = 0.5
overlay = alpha * image.rgb + (1-alpha) * bbox_overlay
overlay = alpha * np.array(img_viz) + (1-alpha) * j.add_rgba_dimension(bbox_overlay)
j.get_rgb_image(overlay).save("bbox.png")

j.setup_visualizer()
j.show_cloud("1",  
    t3d.apply_transform(
        obs_point_cloud_image.reshape(-1,3),
        image.camera_pose
    ) * 3.0
)

j.show_cloud("2",  
    t3d.apply_transform(
        renderer.meshes[obj_idx].vertices,
        image.camera_pose @ pose
    )* 3.0,
    color=j.RED
)

points = np.zeros((9,3))
points[0, :] = np.array([dims[0]/2, -dims[1]/2, dims[2]/2]  )
points[1, :] = np.array([-dims[0]/2, -dims[1]/2, dims[2]/2])
points[2, :] = np.array([-dims[0]/2, dims[1]/2, dims[2]/2])
points[3, :] = np.array([dims[0]/2, dims[1]/2, dims[2]/2])
points[4, :] = np.array([dims[0]/2, -dims[1]/2, -dims[2]/2])
points[5, :] = np.array([-dims[0]/2, -dims[1]/2, -dims[2]/2])
points[6, :] = np.array([-dims[0]/2, dims[1]/2, -dims[2]/2])
points[7, :] = np.array([dims[0]/2, dims[1]/2, -dims[2]/2])
points[8, :] = np.array([0.0, 0.0, 0.0])
new_points = j.t3d.apply_transform(points, image.camera_pose @ pose)

j.show_cloud("3",  
    new_points* 3.0,
    color=j.BLUE
)




from IPython import embed; embed()

