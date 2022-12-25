import matplotlib.pyplot as plt
import numpy as np
import os
import pybullet as p
import pybullet_data
from jax3dp3.viz import make_gif_from_pil_images
from PIL import Image
from copy import copy
from jax3dp3.viz import save_depth_image, get_depth_image, multi_panel
import jax3dp3.utils
import jax3dp3.viz

p.connect(p.DIRECT)
p.resetSimulation()
# p.setGravity(0, 0, -5)

p.setAdditionalSearchPath(pybullet_data.getDataPath())

cubeStartPos = [0, 0, 0.0]
cubeStartOrientation = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
filename_1 = os.path.join(jax3dp3.utils.get_assets_dir(), "models/003_cracker_box/textured_simple.obj")
brick_coll = p.createCollisionShape(p.GEOM_MESH, fileName=filename_1)
brick_vis = p.createVisualShape(p.GEOM_MESH, fileName=filename_1)
brick = p.createMultiBody(baseMass=1110.0001,
                            baseCollisionShapeIndex=brick_coll, baseVisualShapeIndex=brick_vis,
                            basePosition=cubeStartPos,
                            baseOrientation=cubeStartOrientation)

cubeStartPos = [-30, -100, 0.0]
cubeStartOrientation = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
filename_2 = os.path.join(jax3dp3.utils.get_assets_dir(), "models/004_sugar_box/textured_simple.obj")
obj_2_coll = p.createCollisionShape(p.GEOM_MESH, fileName=filename_2)
obj_2_vis = p.createVisualShape(p.GEOM_MESH, fileName=filename_2)
obj_2 = p.createMultiBody(baseMass=0.0001,
                            baseCollisionShapeIndex=obj_2_coll, baseVisualShapeIndex=obj_2_vis,
                            basePosition=cubeStartPos,
                            baseOrientation=cubeStartOrientation)

# p.resetBaseVelocity(sphere_obj, [0.0, 4.0, -1.0], [0.0, 0.0, 0.0])

# p.changeDynamics(planeId, -1, restitution=1.0)
p.changeDynamics(brick, -1, restitution=1.0)
p.changeDynamics(obj_2, -1, restitution=1.0)

viewMatrix = p.computeViewMatrix(
    cameraEyePosition=np.array([1000.0, 0.0, 0.0]),
    cameraTargetPosition=np.array([0.0, 0.0, 0.0]),
    cameraUpVector=np.array([0.0, 0.0, 1.0]),
)

fov = 30.0
width = 640  # 3200
height = 480  # 2400
aspect_ratio = width / height
near = 10.0
far = 2000.0
projMatrix = p.computeProjectionMatrixFOV(fov, aspect_ratio, near, far)


cx, cy = width / 2.0, height / 2.0
fov_y = np.deg2rad(fov)
fov_x = 2 * np.arctan(aspect_ratio * np.tan(fov_y / 2.0))
fx = cx / np.tan(fov_x / 2.0)
fy = cy / np.tan(fov_y / 2.0)

rgb_imgs = []
depth_imgs = []
for y in np.linspace(-200.0, 200.0,70):
    # for _ in range(5):     
    #     p.stepSimulation()
    new_position = copy(cubeStartPos)
    new_position[1] = y
    p.resetBasePositionAndOrientation(obj_2, new_position, cubeStartOrientation)
    w,h, rgb, depth, segmentation = p.getCameraImage(width, height,
        viewMatrix,
        projMatrix
    )
    depth = far * near / (far - (far - near) * depth)
    rgb_imgs.append(rgb)
    depth_imgs.append(depth)

np.savez("data.npz", rgb_imgs=rgb_imgs, depth_imgs=depth_imgs, fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)

images = []
max_depth = 2000.
for (i,rgb) in enumerate(rgb_imgs):
    img = Image.fromarray(
        rgb.astype(np.int8), mode="RGBA"
    )
    depth_img = jax3dp3.viz.get_depth_image(depth_imgs[i],max=max_depth)
    dst = multi_panel([img, depth_img], ["Frame {}".format(i), "Depth"], 0, 100, 40)

    images.append(
        dst
    )
images[0].save(
    fp="rgb.gif",
    format="GIF",
    append_images=images,
    save_all=True,
    duration=100,
    loop=0,
)
