import pybullet as p
import pybullet_data
import imageio
import numpy as np

# Initialize the PyBullet physics simulation
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set up the simulation environment
p.setGravity(0, 0, -10)
plane_id = p.loadURDF("plane.urdf")

# Create the first sphere
sphere_radius1 = 0.5
sphere_mass1 = 1
sphere_position1 = [-2, 0, 1]
sphere_start_velocity1 = [10, 0, 0]
sphere_shape1 = p.createCollisionShape(p.GEOM_SPHERE, radius=sphere_radius1)
sphere_id1 = p.createMultiBody(sphere_mass1, sphere_shape1, basePosition=sphere_position1)
p.resetBaseVelocity(sphere_id1, sphere_start_velocity1)

# Create the second sphere
sphere_radius2 = 0.5
sphere_mass2 = 1
sphere_position2 = [2, 0, 1]
sphere_start_velocity2 = [-10, 0, 0]
sphere_shape2 = p.createCollisionShape(p.GEOM_SPHERE, radius=sphere_radius2)
sphere_id2 = p.createMultiBody(sphere_mass2, sphere_shape2, basePosition=sphere_position2)
p.resetBaseVelocity(sphere_id2, sphere_start_velocity2)

# Array to store frames
frames = []

# Step through the simulation
for i in range(100):
    p.stepSimulation()
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

# Save GIF
imageio.mimsave('simulation.gif', frames, 'GIF', fps=15)

p.disconnect()
