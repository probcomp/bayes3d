import pybullet as p
import pybullet_data
import numpy as np
import open3d as o3d
import trimesh as tm
import bayes3d as b
import imageio
from scipy.spatial.transform import Rotation as R
from PIL import Image

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

def o3d_to_trimesh(mesh):
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    tri_normals = np.asarray(mesh.triangle_normals)
    vert_normals = np.asarray(mesh.vertex_normals)
    mesh = tm.Trimesh(vertices=vertices, faces=faces, vertex_normals=vert_normals, face_normals=tri_normals)
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
    """
    Renders a scene using PyBullet.

    Args:
        scene (Scene): The scene object.

    Returns:
        PIL.Image.Image: The rendered image.
    """
    pyb_sim = PybulletSimulator(camera=scene.camera, floor = scene.floor)
    for body in scene.bodies.values():
        pyb_sim.add_body_to_simulation(body)
    image_rgb, depth, _ = pyb_sim.capture_image()
    pyb_sim.close()
    image = Image.fromarray(image_rgb)
    return image

def create_box(pose, scale = [1,1,1], restitution=1, friction=0, velocity=0, angular_velocity = [0,0,0], id=None):
    """
    Creates a box-shaped Body object.
    """
    position = pose[:3, 3]
    orientation = pose[:3, :3]
    return create_box(position, scale, restitution, friction, velocity, angular_velocity, orientation, id)


def create_box(position, scale=[1,1,1], restitution=1, friction=0, velocity=0, angular_velocity = [0,0,0], orientation=None, id=None):
    """
    Creates a box-shaped Body object.
    """
    path_to_box = "../assets/sample_objs/cube.obj"
    mesh = tm.load(path_to_box)
    pose = np.eye(4)
    pose[:3, :3] = orientation if orientation is not None else np.eye(3)
    pose[:3, 3] = position
    obj_id = "box" if id is None else id
    body = Body(obj_id, pose, mesh, file_dir=path_to_box, restitution=restitution, friction=friction, velocity=velocity, angular_velocity=angular_velocity, scale=scale)
    return body


def create_sphere(position, scale = [1,1,1], velocity=0, angular_velocity = [0,0,0], restitution=1, friction=0, id=None):
    """
    Creates a sphere-shaped Body object.

    Args:
        position (np.ndarray): The position of the sphere.
        radius (float): The radius of the sphere.
        velocity (float, optional): The initial velocity of the sphere. Default is 0.
        restitution (float, optional): The restitution coefficient of the sphere. Default is 1.
        friction (float, optional): The friction coefficient of the sphere. Default is 0.
        id (str, optional): The ID of the sphere. Default is None.

    Returns:
        Body: The created Body object representing the sphere.
    """
    path_to_sphere = "../assets/sample_objs/sphere.obj"
    mesh = tm.load(path_to_sphere)
    pose = np.eye(4)
    pose[:3, 3] = position
    obj_id = "sphere" if id is None else id
    body = Body(obj_id, pose, mesh, file_dir=path_to_sphere, restitution=restitution, friction=friction, velocity=velocity, angular_velocity=angular_velocity, scale = scale)
    return body


def make_body_from_obj_pose(obj_path, pose, velocity=0, angular_velocity = [0,0,0], restitution=1, friction=0, id=None, scale = None ):
    """
    Creates a Body object from an OBJ file with a given pose.

    Args:
        obj_path (str): The path to the OBJ file.
        pose (np.ndarray): The pose of the object.
        velocity (float, optional): The initial velocity of the object. Default is 0.
        restitution (float, optional): The restitution coefficient of the object. Default is 1.
        friction (float, optional): The friction coefficient of the object. Default is 0.
        id (str, optional): The ID of the object. Default is None.

    Returns:
        Body: The created Body object.
    """
    mesh = tm.load(obj_path)
    obj_id = "obj_mesh" if id is None else id
    body = Body(obj_id, pose, mesh, velocity=velocity, angular_velocity = angular_velocity, friction=friction, restitution=restitution, file_dir=obj_path, scale = scale)
    return body


def make_body_from_obj(obj_path, position, friction=0, restitution=1, velocity=0, angular_velocity = [0,0,0], orientation=None, id=None, scale=None):
    """
    Creates a Body object from an OBJ file with a given position.

    Args:
        obj_path (str): The path to the OBJ file.
        position (np.ndarray): The position of the object.
        friction (float, optional): The friction coefficient of the object. Default is 0.
        restitution (float, optional): The restitution coefficient of the object. Default is 1.
        velocity (float, optional): The initial velocity of the object. Default is 0.
        orientation (np.ndarray, optional): The orientation of the object as a rotation matrix. Default is None.
        id (str, optional): The ID of the object. Default is None.

    Returns:
        Body: The created Body object.
    """
    pose = np.eye(4)
    pose[:3, :3] = orientation if orientation is not None else np.eye(3)
    pose[:3, 3] = position
    return make_body_from_obj_pose(obj_path, pose, id=id, friction=friction, restitution=restitution, velocity=velocity, scale=scale, angular_velocity=angular_velocity)

class Body:
    def __init__(self, object_id, pose, mesh, file_dir = None, restitution=0.8, friction=0, damping=0, transparency=1, velocity=[0,0,0], angular_velocity = [0,0,0],mass=1, texture=None, color=[1, 0, 0], scale=None):
        self.id = object_id
        self.pose = pose
        self.restitution = restitution
        self.friction = friction
        self.damping = damping
        self.transparency = transparency
        self.velocity = velocity
        self.angular_velocity = angular_velocity
        self.texture = texture
        self.color = color
        self.mass = mass
        self.file_dir = file_dir
        self.mesh = mesh
        self.scale = scale

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
    
    def get_angular_velocity(self):
        return self.angular_velocity

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
    
    def get_scale(self):
        return self.scale
        
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

    def set_angular_velocity(self, angular_velocity):
        self.angular_velocity = angular_velocity

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

    def set_scale(self, scale):
        self.scale = scale


    # Miscellaneous methods
    def get_fields(self):
        return f"Body ID: {self.id}, Pose: {self.pose}, Restitution: {self.restitution}, Friction: {self.friction}, Damping: {self.damping}, Transparency: {self.transparency}, Velocity: {self.velocity}, Texture: {self.texture}, Color: {self.color}"
    
    def __str__(self):
        return f"Body ID: {self.id}, Position: {self.get_position()}"



class Scene:
    def __init__(self, id = None, bodies=None, camera=None, timestep = 1/60, light=None, gravity = [0,0,0], downsampling = 1, floor = True):
        self.scene_id = id if id is not None else "scene"
        self.bodies = bodies if bodies is not None else {}
        self.gravity = gravity
        self.timestep = timestep
        self.camera = camera
        self.floor = floor
        self.pyb_sim = None
        self.downsampling = downsampling
        print(self.floor)


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
    
    def set_floor(self, floor):
        self.floor = floor
        return self.floor

    def set_camera_position_target(self, position, target):
        self.camera = [position, target]
        return self.camera

    def set_light(self, light: Body):
        self.light = light 
        return self.light
    
    def set_gravity(self, gravity):
        self.gravity = gravity
        return self.gravity
    
    def set_downsampling(self, downsampling):
        self.downsampling = downsampling
        return self.downsampling
    
    def set_timestep(self, timestep):
        self.timestep = timestep
        return self.timestep
    
    def render(self, render_func):
        image = render_func(self)
        return image
    
    def simulate(self, timesteps):
        # create physics simulator 
        pyb = PybulletSimulator(timestep=self.timestep, gravity=self.gravity, camera = self.camera, downsampling = self.downsampling, floor = self.floor )
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
    
class PybulletSimulator(object):
    def __init__(self, timestep=1/60, gravity=[0,0,0], floor_restitution=0.5, camera = None, downsampling=1, floor = True):
        self.timestep = timestep
        self.gravity = gravity
        self.client = p.connect(p.DIRECT)
        self.step_count = 0
        self.frames = [] 
        self.depth = []
        self.pyb_id_to_body_id = {}
        self.body_poses = {}
        self.camera = camera
        self.downsampling = downsampling
        self.floor = floor 

        # Set up the simulation environment
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(self.gravity[0], self.gravity[1], self.gravity[2], physicsClientId=self.client)
        p.setPhysicsEngineParameter(fixedTimeStep=self.timestep, physicsClientId=self.client)
        print(self.floor)
        if self.floor:
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
            p.changeDynamics(self.plane_id, -1, restitution=floor_restitution)
    
    def add_body_to_simulation(self, body):
        """
        Add a body to the pybullet simulation.
        :param body: a Body object.
        :return: None
        """
        # vertices = body.mesh.vertices.tolist()
        # faces = body.mesh.faces.ravel().tolist()
        obj_file_dir = body.file_dir
        mesh_scale = body.scale


        # Create visual and collision shapes
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH, 
                                            # vertices=vertices, 
                                            # indices=faces,
                                            fileName = obj_file_dir,
                                            meshScale = mesh_scale,
                                            physicsClientId=self.client,
                                            rgbaColor=np.append(body.color, body.transparency),
                                            )
        
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                # vertices=vertices, 
                                                # indices=faces,
                                                fileName = obj_file_dir,
                                                meshScale = mesh_scale,
                                                physicsClientId=self.client, 
                                                )

        # Get the orientation matrix
        rot_matrix = body.get_orientation()

        # Convert the rotation matrix to quaternion
        r = R.from_matrix(rot_matrix)
        quaternion = r.as_quat()

        # Create a multibody with the created shapes
        pyb_id = p.createMultiBody(baseMass=body.mass,
                                    baseCollisionShapeIndex=collisionShapeId,
                                    baseVisualShapeIndex=visualShapeId, 
                                    basePosition=body.get_position(),
                                    baseOrientation=quaternion,
                                    physicsClientId=self.client,
                                    )


        # Set physical properties
        p.changeDynamics(pyb_id, -1, restitution=body.restitution, lateralFriction=body.friction, 
                        linearDamping=body.damping, physicsClientId=self.client)

        # Set initial velocity if specified
        if body.velocity != [0,0,0] or body.angular_velocity != [0,0,0]:
            p.resetBaseVelocity(pyb_id, linearVelocity=body.velocity, angularVelocity = body.angular_velocity, physicsClientId=self.client)
# 
        # If texture is specified, load it
        if body.texture is not None:
            textureId = p.loadTexture(body.texture, physicsClientId=self.client)
            p.changeVisualShape(pyb_id, -1, textureUniqueId=textureId, physicsClientId=self.client)

        # Add to mapping from pybullet id to body id
        self.pyb_id_to_body_id[pyb_id] = body.id
        self.body_poses[body.id] = []

    def check_collision(self):
        for body in self.pyb_id_to_body_id.keys():
            collisions = p.getContactPoints(bodyA=body, physicsClientId=self.client)
            if len(collisions) > 0:
                print(f"Body {body} is colliding.")

    def step_simulation(self):
        self.step_count+=1
        p.stepSimulation(physicsClientId=self.client)
    
    def update_body_poses(self):
        for pyb_id in self.pyb_id_to_body_id.keys():
            position, orientation = p.getBasePositionAndOrientation(pyb_id, physicsClientId=self.client)
            orientation = p.getEulerFromQuaternion(orientation)
            pose = np.eye(4)
            pose[:3, :3] = orientation
            pose[:3, 3] = position
            self.body_poses[self.pyb_id_to_body_id[pyb_id]].append(pose)
    
    def simulate(self, steps): 
        # returns frames, poses of objects over time
        for i in range(steps):
            if i % self.downsampling == 0:
                rgb, depth, segm = self.capture_image()
                self.frames.append(rgb)
                # self.depth.append(depth)
                # self.update_body_poses()
            self.step_simulation()

        self.close()

    #TODO; fix this function to return the correct depth image
    def depth_to_image(self, depth_array):
        # Normalize the depth array between 0 and 1 for visualization
        depth_img = (depth_array - np.min(depth_array)) / (np.max(depth_array) - np.min(depth_array))
        depth_uint8 = np.uint8(depth_img * 255)
        return depth_uint8

    def capture_image(self, up_vector=[0, 0, 1], distance=7, yaw=0, pitch=-30, roll=0, dims=[960, 720]):
    # if no position is specified, use arbitrary default position. 
        if self.camera is None:
            view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 0], distance=distance, yaw=yaw, pitch=pitch, roll=roll,
                                                                upAxisIndex=2)
        else: 
            position = self.camera[0]
            target = self.camera[1]
            view_matrix = p.computeViewMatrix(cameraEyePosition=position, cameraTargetPosition=target, cameraUpVector=up_vector)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=float(dims[0]) / dims[1], nearVal=0.1, farVal=100.0)
        width, height, rgb_pixels, depth_pixels, segm_pixels = p.getCameraImage(width=dims[0], height=dims[1], viewMatrix=view_matrix,
                                                projectionMatrix=proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        
        # Convert rgb_pixels (a list of RGBA values) into an array
        rgb_array = np.array(rgb_pixels, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (dims[1], dims[0], 4))
        rgb_array = rgb_array[:, :, :3]  # strip off the alpha channel

        # # Convert depth_pixels (a list of depth values) into an array
        # depth_array = np.array(depth_pixels)
        # depth_array = np.reshape(depth_array, (dims[1], dims[0]))
        # far = 1000.
        # near = 0.01
        # depth = far * near / (far - (far - near) * depth_array)
        # depth_img = self.depth_to_image(depth)
        depth_img = None 

        # Convert segm_pixels (a list of segmentation values) into an array
        # segm_array = np.array(segm_pixels)
        # segm_array = np.reshape(segm_array, (dims[1], dims[0]))
        segm_array = None 

        return rgb_array, depth_img, segm_array
    
    def create_gif(self, path, fps=15):
        imageio.mimsave(path, self.frames, duration = (1000 * (1/fps)))
        
    def set_velocity(self, obj_id):
        return
    
    # adjusts the timestep of the simulation
    def set_timestep(self, dt):
        self.timestep = dt
        p.setPhysicsEngineParameter(fixedTimeStep=self.timestep, physicsClientId=self.client)

    # adjusts the gravity of the simulation
    def set_gravity(self, g):
        self.gravity = g
        p.setGravity(self.gravity[0],self.gravity[1], self.gravity[2], physicsClientId=self.client)

    def close(self):
        p.resetSimulation(physicsClientId=self.client)
        p.disconnect(self.client)
    
    # returns a mapping of body_id to poses over time
    def get_object_poses(self):
        return self.body_poses

    def add_plane(self):
        return
