import pybullet as p
import pybullet_data
import imageio
import numpy as np
import pickle
import os

# Initialize the PyBullet physics simulation
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set up the simulation environment
p.setGravity(0, 0, -10)
plane_id = p.loadURDF("plane.urdf")

# restitution value
restitution = 0.9

# Create the first sphere
sphere_radius1 = 0.5
sphere_mass1 = 1
sphere_position1 = [-2, 0, 1]
sphere_start_velocity1 = [10, 0, 0]
sphere_shape1 = p.createCollisionShape(p.GEOM_SPHERE, radius=sphere_radius1, collisionFramePosition=sphere_position1)
sphere_id1 = p.createMultiBody(sphere_mass1, sphere_shape1, basePosition=sphere_position1, baseInertialFramePosition=sphere_position1)
p.resetBaseVelocity(sphere_id1, sphere_start_velocity1)
p.changeDynamics(sphere_id1, -1, restitution=restitution)

# Create the second sphere
sphere_radius2 = 0.5
sphere_mass2 = 1
sphere_position2 = [2, 0, 1]
sphere_start_velocity2 = [-10, 0, 0]
sphere_shape2 = p.createCollisionShape(p.GEOM_SPHERE, radius=sphere_radius2, collisionFramePosition=sphere_position2)
sphere_id2 = p.createMultiBody(sphere_mass2, sphere_shape2, basePosition=sphere_position2, baseInertialFramePosition=sphere_position2)
p.resetBaseVelocity(sphere_id2, sphere_start_velocity2)
p.changeDynamics(sphere_id2, -1, restitution=restitution)

# Arrays for serialization 
frames = []
sphere_loc = []
sphere_loc2 = []

# Step through the simulation
for i in range(200):
    p.stepSimulation()
    # record positions of spheres
    sphere_position1 = p.getBasePositionAndOrientation(sphere_id1)[0]
    sphere_position2 = p.getBasePositionAndOrientation(sphere_id2)[0]
    sphere_loc.append(sphere_position1)
    sphere_loc2.append(sphere_position2)
    # save a frame every fifth 
    if i % 5 == 0:
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 0], distance=5, yaw=0, pitch=-30, roll=0,
                                                        upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=float(960) / 720, nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960, height=720, viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720, 960, 4))
        rgb_array = rgb_array[:, :, :3]  # remove alpha channel
        frames.append(rgb_array)

# Save GIF and serialize the sphere locations
imageio.mimsave('balls-simulation.gif', frames, 'GIF', duration=1000 * (1/15))
# with open('sph_loc.pkl', 'wb') as f:
#     pickle.dump(sphere_loc, f)
# with open('sph_loc2.pkl', 'wb') as f:
#     pickle.dump(sphere_loc2, f)

p.disconnect()
