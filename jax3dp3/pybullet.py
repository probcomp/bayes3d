import numpy as np
import pybullet as p
import jax.numpy as jnp
import jax3dp3.transforms_3d as t3d
import jax3dp3.bbox
from itertools import product
from collections import defaultdict, deque, namedtuple
import trimesh

NULL_ID = -1
STATIC_MASS = 0
CLIENT = 0

DEFAULT_CLIENT = p


DEFAULT_EXTENTS = [1, 1, 1]
DEFAULT_RADIUS = 0.5
DEFAULT_HEIGHT = 1
DEFAULT_MESH = ''
DEFAULT_SCALE = [1, 1, 1]
DEFAULT_NORMAL = [0, 0, 1]

def plural(word):
    exceptions = {'radius': 'radii'}
    if word in exceptions:
        return exceptions[word]
    if word.endswith('s'):
        return word
    return word + 's'

def get_default_geometry():
    return {
        'halfExtents': DEFAULT_EXTENTS,
        'radius': DEFAULT_RADIUS,
        'length': DEFAULT_HEIGHT, # 'height'
        'fileName': DEFAULT_MESH,
        'meshScale': DEFAULT_SCALE,
        'planeNormal': DEFAULT_NORMAL,
    }

def get_box_geometry(width, length, height):
    return {
        'shapeType': p.GEOM_BOX,
        'halfExtents': [width/2., length/2., height/2.]
    }

def get_sphere_geometry(radius):
    return {
        'shapeType': p.GEOM_SPHERE,
        'radius': radius,
    }

def get_capsule_geometry(radius, height):
    return {
        'shapeType': p.GEOM_CAPSULE,
        'radius': radius,
        'length': height,
    }

def get_plane_geometry(normal):
    return {
        'shapeType': p.GEOM_PLANE,
        'planeNormal': normal,
    }

def get_mesh_geometry(path, scale=1.):
    return {
        'shapeType': p.GEOM_MESH,
        'fileName': path,
        'meshScale': scale*np.ones(3),
    }

def create_shape_array(geoms, poses, colors=None):
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/pybullet.c
    # createCollisionShape: height
    # createVisualShape: length
    # createCollisionShapeArray: lengths
    # createVisualShapeArray: lengths
    poses
    mega_geom = defaultdict(list)
    for geom in geoms:
        extended_geom = get_default_geometry()
        extended_geom.update(geom)
        #extended_geom = geom.copy()
        for key, value in extended_geom.items():
            mega_geom[plural(key)].append(value)

    collision_args = mega_geom.copy()
    for pose in poses:
        collision_args['collisionFramePositions'].append(np.array(pose[:3,3]))
        collision_args['collisionFrameOrientations'].append(np.array(t3d.rotation_matrix_to_xyzw(pose[:3,:3])))
    collision_id = p.createCollisionShapeArray(physicsClientId=CLIENT, **collision_args)
    if (colors is None): # or not has_gui():
        return collision_id, NULL_ID

    visual_args = mega_geom.copy()
    for pose, color in zip(poses, colors):
        # TODO: color doesn't seem to work correctly here
        visual_args['rgbaColors'].append(color)
        visual_args['visualFramePositions'].append(np.array(pose[:3,3]))
        visual_args['visualFrameOrientations'].append(t3d.rotation_matrix_to_xyzw(pose[:3,:3]))
    visual_id = p.createVisualShapeArray(physicsClientId=CLIENT, **visual_args)
    return collision_id, visual_id

def create_body(collision_id=NULL_ID, visual_id=NULL_ID, mass=STATIC_MASS):
    return p.createMultiBody(baseMass=mass, baseCollisionShapeIndex=collision_id,
                             baseVisualShapeIndex=visual_id, physicsClientId=CLIENT)


class Saver(object):
    # TODO: contextlib
    def save(self):
        pass

    def restore(self):
        raise NotImplementedError()

    def __enter__(self):
        # TODO: move the saving to enter?
        self.save()
        # return self

    def __exit__(self, type, value, traceback):
        self.restore()


def get_connection(client=None):
    client = client or DEFAULT_CLIENT
    return client.getConnectionInfo()["connectionMethod"]

def has_gui(client=None):
    client = client or DEFAULT_CLIENT
    return get_connection(client=client) == p.GUI

def set_renderer(enable, client=None):
    client = client or DEFAULT_CLIENT
    if not has_gui(client=client):
        return

    client.configureDebugVisualizer(p.COV_ENABLE_RENDERING, int(enable))

class LockRenderer(Saver):
    # disabling rendering temporary makes adding objects faster
    def __init__(self, client=None, lock=True, **kwargs):
        self.client = client or DEFAULT_CLIENT
        # skip if the visualizer isn't active
        if has_gui(client=self.client) and lock:
            set_renderer(enable=False, client=self.client)

    def restore(self):
        if not has_gui(client=self.client):
            return

        set_renderer(enable=True, client=self.client)

# def load_pybullet(filename, fixed_base=False, scale=1.0, client=None, **kwargs):
#     client = client or DEFAULT_CLIENT
#     with LockRenderer(client=client):
#         if filename.endswith(".obj"):
#             # TODO: fixed_base => mass = 0?
#             body = create_obj(filename, scale=scale, client=client, **kwargs)
#         else:
#             raise ValueError(filename)
#     return body


def get_mesh_geometry(path, scale=1.0):
    return {
        "shapeType": p.GEOM_MESH,
        "fileName": path,
        "meshScale": scale * np.ones(3),
    }


def create_collision_shape(geometry, pose=np.eye(4), client=None, **kwargs):
    # TODO: removeCollisionShape
    # https://github.com/bulletphysics/bullet3/blob/5ae9a15ecac7bc7e71f1ec1b544a55135d7d7e32/examples/pybullet/examples/getClosestPoints.py
    client = client or DEFAULT_CLIENT
    collision_args = {
        "collisionFramePosition": np.array(pose[:3,3]),
        "collisionFrameOrientation": np.array(t3d.rotation_matrix_to_xyzw(pose[:3,:3]))
        #'flags': p.GEOM_FORCE_CONCAVE_TRIMESH,
    }
    collision_args.update(geometry)
    if "length" in collision_args:
        # TODO: pybullet bug visual => length, collision => height
        collision_args["height"] = collision_args["length"]
        del collision_args["length"]
    return client.createCollisionShape(**collision_args)


def create_visual_shape(
    geometry, pose=np.eye(4), color=None, specular=None, client=None, **kwargs
):
    client = client or DEFAULT_CLIENT
    visual_args = {
        "rgbaColor": color,
        "visualFramePosition": np.array(pose[:3,3]),
        "visualFrameOrientation": np.array(t3d.rotation_matrix_to_xyzw(pose[:3,:3])),
    }
    visual_args.update(geometry)
    visual_args["specularColor"] = [0,0,0]
    return client.createVisualShape(**visual_args)


def create_table(
    width,
    length,
    height,
    thickness,
    leg_width,
    **kwargs
):
    surface = get_box_geometry(width, length, thickness)
    surface_pose = t3d.transform_from_pos(jnp.array([0.0, 0.0, height/2.0 - thickness/2.]))

    leg_height = height-thickness
    leg_geometry = get_box_geometry(width=leg_width, length=leg_width, height=leg_height)
    legs = [leg_geometry for _ in range(4)]
    leg_center = np.array([width, length])/2. - leg_width/2.0*np.ones(2)
    leg_xys = [np.multiply(leg_center, np.array(signs))
                for signs in product([-1, +1], repeat=len(leg_center))]
    leg_poses = [t3d.transform_from_pos(jnp.array([x, y, leg_height/2. - height/2.0])) for x, y in leg_xys]

    geoms = [surface] + legs
    poses = [surface_pose] + leg_poses
    color = (0.7, 0.7, 0.7, 1.0)
    colors = [color] + len(legs)*[color]

    collision_id, visual_id = create_shape_array(geoms, poses, colors, **kwargs)
    body = create_body(collision_id, visual_id, **kwargs)
    return body, np.array([width, length, height])
    
def create_obj_centered(
    ycb_path, scale=1.0
):
    mesh = trimesh.load(ycb_path)
    mesh.vertices = mesh.vertices * scale
    dims, pose = jax3dp3.bbox.axis_aligned_bounding_box(mesh.vertices)
    shift = np.array(pose[:3,3])

    visual_geometry = get_mesh_geometry(
        ycb_path, scale=scale
    )  # TODO: randomly transform
    collision_geometry = get_mesh_geometry(ycb_path, scale=scale)

    geometry_pose = np.array(t3d.transform_from_pos(-shift))
    collision_id = create_collision_shape(
        collision_geometry, pose=geometry_pose
    )
    visual_id = create_visual_shape(
        visual_geometry, color=None, pose=geometry_pose
    )
    body = DEFAULT_CLIENT.createMultiBody(
        baseMass=1.0,
        baseCollisionShapeIndex=collision_id,
        baseVisualShapeIndex=visual_id,
        # basePosition=[0., 0., 0.1]
    )
    return body, np.array(dims)

def set_pose(body, pose, client=None):
    client = client or DEFAULT_CLIENT
    client.resetBasePositionAndOrientation(body, np.array(pose[:3,3]), np.array(t3d.rotation_matrix_to_xyzw(pose[:3,:3])))

def capture_image(camera_pose, height,width, fx,fy, cx,cy, near,far):
    opengl_mtx = np.array(
        [
            [2*fx/width, 0.0, (width -2*cx)/width, 0.0],
            [0.0, 2*fy/height, (-height + 2*cy)/height, 0.0],
            [0.0, 0.0, (-far - near) / (far - near), -2.0*far*near/(far-near)],
            [0.0, 0.0, -1.0, 0.0]
        ]
    )
    projMatrix = opengl_mtx

    mat = np.array(t3d.transform_from_rot( t3d.rotation_from_axis_angle(jnp.array([1.0, 0.0, 0.0]),jnp.pi)))
    viewMatrix = np.array(camera_pose).dot(mat)

    _,_, rgb, depth, segmentation = p.getCameraImage(width, height,
        tuple(np.linalg.inv(np.array(viewMatrix)).T.reshape(-1)),
        tuple(np.array(projMatrix).T.reshape(-1)),
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    rgb = np.array(rgb).reshape((height,width,4))
    depth_buffer = np.array(depth).reshape((height,width))
    depth = far * near / (far - (far - near) * depth_buffer)
    segmentation = np.array(segmentation).reshape((height,width))
    return rgb, depth, segmentation

import time

def run_sim():
    p.setRealTimeSimulation(0)
    while True:
        p.stepSimulation()
        time.sleep(0.001)