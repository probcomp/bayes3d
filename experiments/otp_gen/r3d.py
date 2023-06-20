# Interface and DS design for R3D
import pybullet as p
import pybullet_data
import numpy as np
import open3d as o3d
import trimesh as tm
import bayes3d as b
import os 
from PIL import Image
from pybullet_sim import PybulletSimulator

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

# converts o3d triangle_mesh to trimesh mesh
def o3d_to_trimesh(mesh):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    mesh = tm.Trimesh(vertices=vertices, faces=faces, process=False)
    return mesh

def o3d_render(scene): 
    intrinsics = o3d.camera.PinholeCameraIntrinsic()
    renderer = b.o3d_viz.O3DVis(intrinsics=intrinsics) 
    for body in scene.bodies.values():
        mesh = body.mesh
        pose = body.pose
        renderer.render_mesh(mesh, pose)
    camera_pose = scene.camera.pose
    image = renderer.render(camera_pose)
    return image

def pybullet_render(scene):
    pyb_sim = PybulletSimulator()
    for body in scene.bodies.values():
        pyb_sim.add_body_to_simulation(body)
    image_rgb = pyb_sim.capture_image()
    pyb_sim.close()
    image = Image.fromarray(image_rgb)
    return image

def blender_to_pybullet_position(blender_position):
    return blender_position[:]

def blender_to_pybullet_orientation(blender_orientation):
    euler = blender_orientation.to_euler()
    return euler[:]  # Return as a list

def create_box(pose, length, width, height, id = None): 
    position = pose[:3, 3]
    orientation = pose[:3, :3]
    return create_box(position, length, width, height, orientation, id)

def create_box(position, length, width, height, orientation = None, id = None):
    box = o3d.geometry.TriangleMesh.create_box(width=length, height=width, depth=height)
    mesh = o3d_to_trimesh(box)
    orientation = orientation if orientation is not None else np.eye(3)
    pose = np.eye(4)
    pose[:3, :3] = orientation
    pose[:3, 3] = position
    obj_id = "box" if id is None else id
    body = Body(obj_id, pose, mesh)
    return body

def create_sphere(position, radius, id = None):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh = o3d_to_trimesh(sphere)
    pose = np.eye(4)
    pose[:3, 3] = position
    obj_id = "sphere" if id is None else id
    body = Body(obj_id, pose, mesh)
    return body

def make_body_from_obj_pose(obj_path, pose, id=None):
    mesh = tm.exchange.load.load_mesh(obj_path, file_type="obj")
    obj_id = "obj_mesh" if id is None else id
    body = Body(obj_id, pose, mesh)
    return body

def make_body_from_obj(obj_path, position, orientation=None, id=None):
    pose = np.eye(4)
    pose[:3, :3] = orientation if orientation is not None else np.eye(3)
    pose[:3, 3] = position
    return make_body_from_obj_pose(obj_path, pose, id)

class Body:
    def __init__(self, object_id, pose, mesh, restitution=1.0, friction=0, damping=0, transparency=1, velocity=0, mass=1, texture=None, color=[1, 0, 0]):
        self.id = object_id
        self.pose = pose
        self.restitution = restitution
        self.friction = friction
        self.damping = damping
        self.transparency = transparency
        self.velocity = velocity
        self.texture = texture
        self.color = color
        self.mass = mass
        self.mesh = mesh  # trimesh mesh, with vertices and faces
    
    # Getter methods
    def get_id(self):
        return self.id

    def get_pose(self):
        return self.pose

    def get_mesh(self):
        return self.mesh

    def get_restitution(self):
        return self.restitution

    def get_friction(self):
        return self.friction

    def get_damping(self):
        return self.damping

    def get_transparency(self):
        return self.transparency

    def get_velocity(self):
        return self.velocity

    def get_mass(self):
        return self.mass

    def get_texture(self):
        return self.texture

    def get_color(self):
        return self.color

    def get_position(self):
        return self.pose[:3, 3]

    def get_orientation(self):
        return self.pose[:3, :3]
        
    # Setter methods
    def set_id(self, object_id):
        self.id = object_id

    def set_pose(self, pose):
        self.pose = pose

    def set_restitution(self, restitution):
        self.restitution = restitution

    def set_friction(self, friction):
        self.friction = friction

    def set_damping(self, damping):
        self.damping = damping

    def set_transparency(self, transparency):
        self.transparency = transparency

    def set_velocity(self, velocity):
        self.velocity = velocity

    def set_mass(self, mass):
        self.mass = mass

    def set_texture(self, texture):
        self.texture = texture

    def set_color(self, color):
        self.color = color

    def set_position(self, position):
        self.pose[:3, 3] = position

    def set_orientation(self, orientation):
        self.pose[:3, :3] = orientation

    # Miscellaneous methods
    def get_fields(self):
        return f"Body ID: {self.id}, Pose: {self.pose}, Restitution: {self.restitution}, Friction: {self.friction}, Damping: {self.damping}, Transparency: {self.transparency}, Velocity: {self.velocity}, Texture: {self.texture}, Color: {self.color}"
    
    def __str__(self):
        return f"Body ID: {self.id}, Position: {self.get_position()}"



class Scene:
    def __init__(self, id = None, bodies=None, camera=None, light=None, gravity = 0):
        self.scene_id = id if id is not None else "scene"
        self.bodies = bodies if bodies is not None else {}
        self.gravity = 0
        self.camera = camera if camera is not None else Body("camera", np.eye(4), None) #todo 
        self.pyb_sim = None


    def add_body(self, body: Body):
        self.bodies[body.id] = body
        return self.bodies
    
    def add_bodies(self, bodies: list):
        for body in bodies:
            self.add_body(body)
        return self.bodies

    def remove_body(self, body_id):
        if body_id not in self.bodies:
            raise ValueError("Body not in scene")
        else:
            del self.bodies[body_id]
        return self.bodies
    
    def remove_bodies(self, body_ids):
        for body_id in body_ids:
            self.remove_body(body_id)
            print('removed body: ', body_id)
        return self.bodies
    
    def get_bodies(self):
        return self.bodies

    def set_camera(self, camera):
        self.camera = camera
        return self.camera

    def set_light(self, light: Body):
        self.light = light 
        return self.light
    
    def set_gravity(self, gravity):
        self.gravity = gravity
        return self.gravity
    
    def render(self, render_func):
        image = render_func(self)
        return image
    
    def simulate(self, timesteps):
        # create physics simulator 
        pyb = PybulletSimulator()
        self.pyb_sim = pyb

        # add bodies to physics simulator
        for body in self.bodies.values():
            pyb.add_body_to_simulation(body)

        # simulate for timesteps
        pyb.simulate(timesteps)
        # returns pybullet simulation, which you can obtain a gif, poses from. 
        
        return pyb

    def close(self): 
        if self.pyb_sim == None: 
            raise ValueError("No pybullet simulation to close")
        else:
            p.disconnect(self.pyb_sim.client)

    def __str__(self):
        body_str = "\n".join(["    " + str(body) for body in self.bodies.values()])
        return f"Scene ID: {self.scene_id}\nBodies:\n{body_str}"
