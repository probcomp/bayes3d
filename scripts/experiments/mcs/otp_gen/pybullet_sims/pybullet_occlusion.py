import pybullet as p
import pybullet_data
import imageio
import numpy as np
import pickle
import os
print(os.getcwd(), "this is the current working directory")

# Initialize the PyBullet physics simulation
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set up the simulation environment
p.setGravity(0, 0, 0) # Zero gravity 
plane_id = p.loadURDF("plane.urdf")

# Create the wall
wall_half_extents = [1, 0.2, 1]
wall_position = [0, -1, 2]
wall_orientation = p.getQuaternionFromEuler([0, 0, 0])
wall_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=wall_half_extents)
p.createMultiBody(0, wall_id, basePosition=wall_position, baseOrientation=wall_orientation)

# Create the sphere
sphere_radius = 0.5
sphere_mass = 1
sphere_position = [-5, -3, 1]
sphere_start_velocity = [15, 0, 0]
sphere_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=sphere_radius)
sphere_id = p.createMultiBody(sphere_mass, sphere_shape, basePosition=sphere_position)
p.resetBaseVelocity(sphere_id, sphere_start_velocity)

# Arrays for serialization
frames = []
sphere_loc = []
wall_loc = []

# Step through the simulation
for i in range(150):
    p.stepSimulation()
    # record positions of spheres
    sphere_position = p.getBasePositionAndOrientation(sphere_id)[0]
    wall_position = p.getBasePositionAndOrientation(wall_id)[0]
    sphere_loc.append(sphere_position)
    wall_loc.append(wall_position)
    # save a frame every fifth
    if i % 5 == 0:
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 0], distance=5, yaw=0, pitch=-30, roll=0,
                                                        upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=float(960) / 720, nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960, height=720, viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720, 960, 4))
        rgb_array = rgb_array[:, :, :3]
        frames.append(rgb_array)

# Serialize the sphere locations and save gif 
with open('sph_loc.pkl', 'wb') as f:
    pickle.dump(sphere_loc, f)
with open('wall_loc.pkl', 'wb') as f:
    pickle.dump(wall_loc, f)
imageio.mimsave('pybullet_occlusion.gif', frames, duration = (1000 * (1/15)))

p.disconnect()
