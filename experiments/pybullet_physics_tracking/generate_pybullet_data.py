import matplotlib.pyplot as plt
import numpy as np
import os
import pybullet as p
import pybullet_data
from jax3dp3.viz.gif import make_gif
from PIL import Image
from jax3dp3.viz.img import save_depth_image, get_depth_image, multi_panel

p.connect(p.DIRECT)
p.resetSimulation()
p.setGravity(0, 0, -5)

p.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = p.loadURDF("plane.urdf")


cubeStartPos = [-5, -4, 4]
cubeStartOrientation = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
filename_1 = '/home/nishadgothoskar/jax3dp3/experiments/pybullet_physics_tracking/003_cracker_box/textured.obj'
scale_1 = [10.0, 10.0, 10.0]
sphere_collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=0.3)
sphere_visual_shape = p.createVisualShape(p.GEOM_SPHERE,radius=0.3)
sphere_obj = p.createMultiBody(baseMass=1110.0001,
                            baseCollisionShapeIndex=sphere_collision_shape, baseVisualShapeIndex=sphere_visual_shape,
                            basePosition=cubeStartPos,
                            baseOrientation=cubeStartOrientation)

cubeStartPos = [2, 0, 2.1]
cubeStartOrientation = p.getQuaternionFromEuler([0.0, 0.0, 0.0])
filename_2 = '/home/nishadgothoskar/jax3dp3/experiments/pybullet_physics_tracking/003_cracker_box/textured.obj'
scale_2 = [1000.0, 1000.0, 1000.0]
box_collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=1.0)
box_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=1.0)
box_obj = p.createMultiBody(baseMass=0.1,
                            baseCollisionShapeIndex=box_collision_shape, baseVisualShapeIndex=box_visual_shape,
                            basePosition=cubeStartPos,
                            baseOrientation=cubeStartOrientation)

p.resetBaseVelocity(sphere_obj, [0.0, 4.0, -1.0], [0.0, 0.0, 0.0])

p.changeDynamics(planeId, -1, restitution=1.0)
p.changeDynamics(sphere_obj, -1, restitution=1.1)
p.changeDynamics(box_obj, -1, restitution=1.0)

viewMatrix = p.computeViewMatrix(
    cameraEyePosition=np.array([10.0, 0.0, 1.0]),
    cameraTargetPosition=np.array([0.0, 0.0, 1.0]),
    cameraUpVector=np.array([0.0, 0.0, 1.0]),
)

fov = 30.0
width = 640  # 3200
height = 480  # 2400
aspect_ratio = width / height
near = 0.001
far = 200.0
projMatrix = p.computeProjectionMatrixFOV(fov, aspect_ratio, near, far)


cx, cy = width / 2.0, height / 2.0
fov_y = np.deg2rad(fov)
fov_x = 2 * np.arctan(aspect_ratio * np.tan(fov_y / 2.0))
fx = cx / np.tan(fov_x / 2.0)
fy = cy / np.tan(fov_y / 2.0)

rgb_imgs = []
depth_imgs = []
for _ in range(50):
    for _ in range(20):     
        p.stepSimulation()
    w,h, rgb, depth, segmentation = p.getCameraImage(width, height,
        viewMatrix,
        projMatrix
    )
    depth = far * near / (far - (far - near) * depth)
    rgb_imgs.append(rgb)
    depth_imgs.append(depth)

np.savez("data.npz", rgb_imgs=rgb_imgs, depth_imgs=depth_imgs, fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height)


images = []
for (i,rgb) in enumerate(rgb_imgs):
    img = Image.fromarray(
        rgb.astype(np.int8), mode="RGBA"
    )
    dst = multi_panel([img], ["Frame {}".format(i)], 0, 100, 40)

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
