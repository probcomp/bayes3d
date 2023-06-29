import pybullet as p
import numpy as np
import pickle
import os
import open3d as o3d
from pybullet_utils import transformations as tf

def open3d_to_pybullet(point):
    # Convert Open3D coordinate to PyBullet coordinate
    return np.array([point[0], point[2], -point[1]])

def pybullet_to_open3d(point):
    # Convert PyBullet coordinate to Open3D coordinate
    return np.array([point[0], -point[2], point[1]])

def open3d_to_pybullet_pose(pose):
    # Convert Open3D pose (4x4 transformation matrix) to PyBullet pose
    translation = pose[:3, 3]
    rotation = tf.quaternion_from_matrix(pose)
    return translation, rotation

def pybullet_to_open3d_pose(translation, rotation):
    # Convert PyBullet pose (translation and rotation) to Open3D pose (4x4 transformation matrix)
    pose = tf.quaternion_matrix(rotation)
    pose[:3, 3] = translation
    return pose

# create sphere with defined parameters
def create_sphere(radius, mass, position, start_velocity):
    sphere_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)
    sphere_id = p.createMultiBody(mass, sphere_shape, basePosition=position)
    p.resetBaseVelocity(sphere_id, start_velocity)
    return sphere_id

# create rectangle with defined parameters
def create_rectangle(half_extents, position, orientation, start_velocity):
    rectangle_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
    p.createMultiBody(0, rectangle_id, basePosition=position, baseOrientation=orientation)
    p.resetBaseVelocity(rectangle_id, start_velocity)
    return rectangle_id

# create plane with defined parameters
def create_cylinder(radius, height, mass, position, orientation, start_velocity):
    cylinder_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=height)
    cylinder_id = p.createMultiBody(mass, cylinder_shape, basePosition=position, baseOrientation=orientation)
    p.resetBaseVelocity(cylinder_id, start_velocity)
    return cylinder_id

# create plane with defined parameters
def create_plane(position, orientation):
    plane_id = p.createCollisionShape(p.GEOM_PLANE)
    p.createMultiBody(0, plane_id, basePosition=position, baseOrientation=orientation)
    return plane_id

# o3d version
def create_sphere_geom(position, radius):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.compute_vertex_normals()
    sphere.translate(position, relative=False)
    return sphere

def create_cube_geom(position, half_extents):
    cube = o3d.geometry.TriangleMesh.create_box(width=half_extents[0]*2, height=half_extents[1]*2, depth=half_extents[2]*2)
    cube.compute_vertex_normals()
    cube.translate(position, relative=False)
    return cube

def draw_scene(sphere_data, wall_data, sphere_color=[0, 0, 1], wall_color=[1, 0, 0]):
    sphere = create_sphere_geom(sphere_data['positions'], sphere_data['radius'])
    sphere.paint_uniform_color(sphere_color)
    wall = create_cube_geom(wall_data['positions'], wall_data['half_extents'])
    wall.paint_uniform_color(wall_color)
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=960, height=720, visible=False)  # Create an invisible window
    vis.add_geometry(sphere)
    vis.add_geometry(wall)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])  # Set background color
    # Capture the screen (no need to run the visualizer)
    image = vis.capture_screen_float_buffer(do_render=False)
    vis.destroy_window()
    # Convert the floating point buffer to an image
    image = np.array(image)
    image *= 255
    image = image.astype(np.uint8)
    return image


