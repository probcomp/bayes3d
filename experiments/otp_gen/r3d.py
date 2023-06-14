# Interface and DS design for R3D
import pybullet as p
import numpy as np
import open3d as o3d
from pybullet_utils import transformations as tf
# import mathutils

def o3d_to_pybullet_position(o3d_position):
    return np.array(o3d_position)

def o3d_to_pybullet_pose(o3d_pose):
    pybullet_pose = o3d_pose
    pybullet_pose[3, :3] = -pybullet_pose[3, :3]  # Convert rotation from Open3D to PyBullet
    return pybullet_pose

def pybullet_to_o3d_position(pybullet_position):
    return pybullet_position

def pybullet_to_o3d_pose(pybullet_pose):
    o3d_pose = pybullet_pose
    o3d_pose[3, :3] = -o3d_pose[3, :3]  # Convert rotation from PyBullet to Open3D
    return o3d_pose

# def pybullet_to_blender_position(pybullet_position):
#     return mathutils.Vector(pybullet_position)

# def pybullet_to_blender_orientation(pybullet_orientation):
#     euler = mathutils.Euler(pybullet_orientation)
#     return euler.to_quaternion()

def blender_to_pybullet_position(blender_position):
    return blender_position[:]

def blender_to_pybullet_orientation(blender_orientation):
    euler = blender_orientation.to_euler()
    return euler[:]  # Return as a list

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

def o3d_render(scene):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=960, height=720, visible=False)
    # create camera, lighting, background and floor

    for object in scene.bodies:
        if type(object) == Rectangle:
            cube = create_cube_geom(object.position, object.half_extents)
            paint_color = object.color.append(object.transparency)
            cube.paint_uniform_color(paint_color)
            vis.add_geometry(cube)
        elif type(object) == Sphere:
            sphere = create_sphere_geom(object.position, object.radius)
            paint_color = object.color.append(object.transparency)
            sphere.paint_uniform_color(paint_color)
            vis.add_geometry(sphere)
        else:
            raise ValueError("Object type not supported")
        
    # render scene
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1, 1, 1])  # Set background color
    image = vis.capture_screen_float_buffer(do_render=False)
    vis.destroy_window()
    return image

# Base class
class Body:
    def __init__(self, object_id, position, orientation, transparency = 0, velocity = 0, texture = None, color = None):
        self.object_id = object_id
        self.position = position
        self.orientation = orientation
        self.transparency = transparency
        self.velocity = velocity
        self.texture = texture
        self.color = color

    # set transparency
    def set_transparency(self, transparency):
        self.transparency = transparency
        return self.transparency
    
    # set pose
    def set_pose(self, pose):
        self.pose = pose
        return self.pose
    
    # set velocity
    def set_velocity(self, velocity):
        self.velocity = velocity
        return self.velocity


# Class for spheres
class Sphere(Body):
    def __init__(self, object_id, pose, radius, transparency = 0, velocity = 0, texture = None, color = None):
        super().__init__(object_id, pose, transparency, velocity, texture, color)
        self.radius = radius
        self.color = [0, 0, 1]

# Class for rectangles
class Rectangle(Body):
    def __init__(self, object_id, pose, half_extents, transparency = 0, velocity = 0, texture = None, color = None):
        super().__init__(object_id, pose, transparency, velocity, texture, color)
        self.half_extents = half_extents
        self.color = [1, 0, 0]

# Class for meshes
class Mesh(Body):
    def __init__(self, object_id, pose, mesh, transparency = 0, velocity = 0, texture = None, color = None):
        super().__init__(object_id, pose, transparency, velocity, texture, color)
        self.mesh = mesh
    
# Class for scenes
class Scene:
    def __init__(self, scene_id, bodies = {}, camera = None, light = None):
        self.scene_id = scene_id
        self.bodies = bodies
        self.camera = camera

    # add object to scene
    def add_body(self, body: Body):
        self.bodies[body.id] = body
        return self.bodies
    
    # remove object from scene using the object id, otherwise raise an error 
    def remove_body(self, body_id):
        if body_id not in self.bodies:
            raise ValueError("Body not in scene")
        else:
            del self.bodies[body_id]
        return self.bodies
    
    # set camera in a scene
    def set_camera(self, camera):
        self.camera = camera
        return self.camera
    
    # set light in a scene using pose information from a body 
    def set_light(self, light: Body):
        self.light = light 
        return self.light
    
    # render scene using open3d, pybullet, or kubric if camera and light are set
    def render(self, renderer):
        image = renderer.render(self)
        return image
    
    # simulate scene using pybullet, save gif using open3d, pybullet, or kubric
    def simulate(self, renderer, timesteps):
        raise NotImplementedError("Simulation not implemented yet")