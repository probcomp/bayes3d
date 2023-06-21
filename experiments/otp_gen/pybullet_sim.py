import pybullet as p
import pybullet_data
import numpy as np
import open3d as o3d
import trimesh as tm
import bayes3d as b
import os 
import imageio
from scipy.spatial.transform import Rotation as R
from PIL import Image

class PybulletSimulator(object):
    def __init__(self, timestep=1/60, gravity=-10):
        self.timestep = timestep
        self.gravity = gravity
        self.client = p.connect(p.DIRECT)
        self.step_count = 0
        self.frames = [] 
        self.pyb_id_to_body_id = {}
        self.body_poses = {}

        # Set up the simulation environment
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, self.gravity, physicsClientId=self.client)
        p.setPhysicsEngineParameter(fixedTimeStep=self.timestep, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
    
    def add_body_to_simulation(self, body):
        """
        Add a body to the pybullet simulation.
        :param body: a Body object.
        :return: None
        """

        # Get vertices and faces from the trimesh
        vertices = body.mesh.vertices.tolist()
        faces = body.mesh.faces.ravel().tolist()
        # normals = body.mesh.face_normals.ravel().tolist()

        # Create visual and collision shapes
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH, 
                                            vertices=vertices, 
                                            indices=faces,
                                            # normals=normals,
                                            # fileName = "sphere.obj",
                                            physicsClientId=self.client,
                                            rgbaColor=np.append(body.color, body.transparency)
                                            )
        
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                                vertices=vertices, 
                                                indices=faces,
                                                # normals=normals,
                                                # fileName = "sphere.obj",
                                                physicsClientId=self.client, 
                                                # collisionFramePosition=body.get_position()
                                                )

        # Get the orientation matrix
        rot_matrix = body.get_orientation()

        # Convert the rotation matrix to quaternion
        r = R.from_matrix(rot_matrix)
        quaternion = r.as_quat()

        # Create a multibody with the created shapes
        pyb_id = p.createMultiBody(baseMass=1,
                                    baseCollisionShapeIndex=collisionShapeId,
                                    baseVisualShapeIndex=visualShapeId,
                                    basePosition=body.get_position(),
                                    # baseOrientation=quaternion,
                                    physicsClientId=self.client,
                                    baseInertialFramePosition=body.get_position())


        # Set physical properties
        p.changeDynamics(pyb_id, -1, restitution=body.restitution, lateralFriction=body.friction, 
                        linearDamping=body.damping, physicsClientId=self.client)

        # Set initial velocity if specified
        if body.velocity != 0:
            p.resetBaseVelocity(pyb_id, linearVelocity=body.velocity, physicsClientId=self.client)

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
        self.check_collision()
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
            self.frames.append(self.capture_image())
            # self.update_body_poses()
            self.step_simulation()

    def capture_image(self):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[0, 0, 0], distance=5, yaw=0, pitch=-30, roll=0,
                                                            upAxisIndex=2)
        proj_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=float(960) / 720, nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = p.getCameraImage(width=960, height=720, viewMatrix=view_matrix,
                                            projectionMatrix=proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720, 960, 4))
        rgb_array = rgb_array[:, :, :3]
        return rgb_array
    
    def create_gif(self, path, fps):
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
        p.setGravity(0, 0, self.gravity, physicsClientId=self.client)

    def close(self):
        p.resetSimulation(physicsClientId=self.client)
        p.disconnect(self.client)
    
    # returns a mapping of body_id to poses over time
    def get_object_poses(self):
        return self.body_poses

    def add_plane(self):
        return
    
#### Test

# sim = PybulletSimulator()
# sim.add_object("sphere.obj", pose_in_air)
# sim.add_object("wall.obj", pose_in_middle)

# object_poses_over_time = []
# for i in range(100):
#     sim.step_simulation()
#     object_poses_over_time.append(sim.get_object_poses())