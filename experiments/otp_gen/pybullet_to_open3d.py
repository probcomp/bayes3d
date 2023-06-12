import pybullet as p
import pybullet_data
import imageio
import os
from pyb_gen import *
import open3d as o3d
import numpy as np

# print present working directory
print(os.getcwd())

# 1. Pybullet Simulation w/ saved data (positions and orientations)
#     a. Try with simple spheres first 
#     b. Then try with more complex objects meshes, imported from meshes 
# 2. Save gif from pybullet 
# 3. create step by step sim using open3d 
# 4. Save gif from open3d

# create gif using o3d




def create_gif(object_data, save_path='open3d_animation.gif'):
    frames = []
    pos = 0 
    for sphere_pos, wall_pos in zip(object_data['sphere']['positions'], object_data['wall']['positions']):
        # only save every fifth frame
        pos +=1 
        print(pos, "position number")
        if pos % 5 == 0:
            sphere_data = object_data['sphere'].copy()
            wall_data = object_data['wall'].copy()
            sphere_data['positions'] = sphere_pos
            wall_data['positions'] = wall_pos
            image = draw_scene(sphere_data, wall_data)
            frames.append(image)
            print('frame added')
    imageio.mimsave(save_path, frames, duration = (1000 * (1/15)))

# function for returning object data from pybullet, returns object data 
def physics_sim(vis=False):
    p.connect(p.GUI if vis else p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Set up the simulation environment
    p.setGravity(0, 0, 0) # Zero gravity 
    plane_id = p.loadURDF("plane.urdf") #load ground plane 

    # Create the wall parameters 
    wall_half_extents = [1, 0.2, 1]
    wall_position = [0, -1, 2]
    wall_orientation = p.getQuaternionFromEuler([0, 0, 0])
    wall_id = create_rectangle(wall_half_extents, wall_position, wall_orientation, [0, 0, 0])

    # Create the sphere
    sphere_radius = 0.5
    sphere_mass = 1
    sphere_position = [-5, -3, 1]
    sphere_start_velocity = [15, 0, 0]
    sphere_id = create_sphere(sphere_radius, sphere_mass, sphere_position, sphere_start_velocity)

    # Arrays for serialization
    frames = []
    sphere_loc = []
    wall_loc = []
    sphere_orien = []
    wall_orien = []

    sphere_data = {"mesh": 'sphere',
                   "radius": sphere_radius
                  }
    wall_data = {"mesh": 'cube',
                    "half_extents": wall_half_extents
                }
    
    # Step through the simulation
    for i in range(100):
        p.stepSimulation()
        # record positions and orientations
        sphere_position, sphere_orientation = p.getBasePositionAndOrientation(sphere_id)
        wall_position, wall_orientation = p.getBasePositionAndOrientation(wall_id)
        sphere_loc.append(sphere_position)
        wall_loc.append(wall_position)
        sphere_orien.append(sphere_orientation)
        wall_orien.append(wall_orientation)
        # save a frame every fifth
        if vis and i % 5 == 0:
            view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 0], distance=5, yaw=0, pitch=-30, roll=0,
                                                            upAxisIndex=2)
            proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=float(960) / 720, nearVal=0.1, farVal=100.0)
            (_, _, px, _, _) = p.getCameraImage(width=960, height=720, viewMatrix=view_matrix,
                                                projectionMatrix=proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
            rgb_array = np.array(px, dtype=np.uint8)
            rgb_array = np.reshape(rgb_array, (720, 960, 4))
            rgb_array = rgb_array[:, :, :3]
            frames.append(rgb_array)

    if vis:
        imageio.mimsave('pybullet_occlusion.gif', frames, duration = (1000 * (1/15)))
    p.disconnect()

    # store object info in a set 
    sphere_data["positions"] = sphere_loc
    sphere_data["orientations"] = sphere_orien
    wall_data["positions"] = wall_loc
    wall_data["orientations"] = wall_orien
    object_data = {"sphere": sphere_data, "wall": wall_data}
    return object_data

if __name__ == '__main__':
    object_data = physics_sim(vis=True)
    create_gif(object_data)