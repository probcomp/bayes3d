import pybullet as p
import numpy as np
import pickle
import os
import open3d as o3d

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
def create_sphere(position, radius):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sphere.translate(position)
    return sphere

def create_cube(position, half_extents):
    cube = o3d.geometry.TriangleMesh.create_box(width=half_extents[0]*2, height=half_extents[1]*2, depth=half_extents[2]*2)
    cube.translate(position)
    return cube

#TODO: refactor to have more realistic lighitng 
def draw_scene(sphere_data, wall_data, sphere_color=[0, 0, 1], wall_color=[1, 0, 0]):
    sphere = create_sphere(sphere_data['positions'], sphere_data['radius'])
    sphere.paint_uniform_color(sphere_color)
    wall = create_cube(wall_data['positions'], wall_data['half_extents'])
    wall.paint_uniform_color(wall_color)
    o3d.visualization.draw_geometries([sphere, wall])


