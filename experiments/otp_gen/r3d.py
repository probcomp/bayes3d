# Interface and DS design for R3D
import pybullet as p
import numpy as np
import open3d as o3d
import trimesh as tm
import bayes3d as b

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

# wrapper for creating boxes with pose 
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

class Body:
    def __init__(self, object_id, pose, mesh, restitution=1.0, friction=0, damping=0, transparency=0, velocity=0, texture=None, color=None):
        self.id = object_id
        self.pose = pose  # use pose instead of position and orientation
        self.restitution = restitution
        self.friction = friction
        self.damping = damping
        self.transparency = transparency
        self.velocity = velocity
        self.texture = texture
        self.color = color
        self.mesh = mesh

    def set_transparency(self, transparency):
        self.transparency = transparency
        return self.transparency

    def set_pose(self, pose):
        self.pose = pose
        return self.pose
    
    def get_position(self):
        return self.pose[:3, 3]
    
    def get_orientation(self):
        return self.pose[:3, :3]

    def set_velocity(self, velocity):
        self.velocity = velocity
        return self.velocity

    def set_restitution(self, restitution):
        self.restitution = restitution
        return self.restitution

    def set_friction(self, friction):
        self.friction = friction
        return self.friction

    def set_damping(self, damping):
        self.damping = damping
        return self.damping

    def get_fields(self):
        return f"Body ID: {self.id}, Pose: {self.pose}, Restitution: {self.restitution}, Friction: {self.friction}, Damping: {self.damping}, Transparency: {self.transparency}, Velocity: {self.velocity}, Texture: {self.texture}, Color: {self.color}"
    
    def __str__(self):
        return f"Body ID: {self.id}, Position: {self.get_position()}"


class Scene:
    def __init__(self, scene_id = None, bodies={}, camera=None, light=None):
        self.scene_id = scene_id if scene_id is not None else "scene"
        self.bodies = bodies
        self.camera = camera if camera is not None else Body("camera", np.eye(4), None) #todo 

    def add_body(self, body: Body):
        self.bodies[body.id] = body
        return self.bodies

    def remove_body(self, body_id):
        if body_id not in self.bodies:
            raise ValueError("Body not in scene")
        else:
            del self.bodies[body_id]
        return self.bodies

    def set_camera(self, camera):
        self.camera = camera
        return self.camera

    def set_light(self, light: Body):
        self.light = light 
        return self.light
    
    def render(self, render_func):
        image = render_func(self)
        return image
    
    def simulate(self, renderer, timesteps):
        raise NotImplementedError("Simulation not implemented yet")

    def __str__(self):
        body_str = "\n".join(["    " + str(body) for body in self.bodies.values()])
        return f"Scene ID: {self.scene_id}\nBodies:\n{body_str}"
