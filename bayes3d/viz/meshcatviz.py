import meshcat
import numpy as np
import meshcat.geometry as g
from matplotlib.colors import rgb2hex
import bayes3d.transforms_3d as t3d
import jax.numpy as jnp


RED = np.array([1.0, 0.0, 0.0])
GREEN = np.array([0.0, 1.0, 0.0])
BLUE = np.array([0.0, 0.0, 1.0])

VISUALIZER = None

def setup_visualizer():
    global VISUALIZER
    VISUALIZER = meshcat.Visualizer()
    set_background_color([1, 1, 1])

def get_visualizer():
    global VISUALIZER
    return VISUALIZER

def set_background_color(color):
    VISUALIZER["/Background"].set_property("top_color", color)
    VISUALIZER["/Background"].set_property("bottom_color", color)

def clear():
    global VISUALIZER
    VISUALIZER.delete()

def set_pose(channel, pose):
    VISUALIZER[channel].set_transform(np.array(pose,dtype=np.float64))

def show_cloud(channel, point_cloud, color=None, size=0.01):
    global VISUALIZER
    if len(point_cloud.shape) == 3:
        point_cloud = t3d.point_cloud_image_to_points(point_cloud)
    point_cloud = np.transpose(np.array(point_cloud))
    if color is None:
        color = np.zeros_like(point_cloud)
    elif len(color.shape) == 1:
        color = np.tile(color.reshape(-1,1), (1,point_cloud.shape[1]))
    color = np.array(color)
    obj = g.PointCloud(point_cloud, color, size=size)
    VISUALIZER[channel].set_object(obj)


def show_trimesh(channel, mesh, color=None, wireframe=False, opacity=1.0):
    global VISUALIZER
    if color is None:
        color = [1, 0, 0]
    material = g.MeshLambertMaterial(color=int(rgb2hex(color)[1:],16), wireframe=wireframe, opacity=opacity)
    obj = g.TriangularMeshGeometry(mesh.vertices, mesh.faces)
    VISUALIZER[channel].set_object(obj, material)


def show_pose(channel, pose, size=0.1):
    global VISUALIZER
    pose_x = t3d.transform_from_pos(jnp.array([size/2.0, 0.0, 0.0]))
    objx = g.Box(np.array([size, size/10.0, size/10.0]))
    matx = g.MeshLambertMaterial(color=0xf41515,
                                reflectivity=0.8)

    pose_y = t3d.transform_from_pos(jnp.array([0.0, size/2.0, 0.0]))
    objy = g.Box(np.array([size/10.0, size, size/10.0]))
    maty = g.MeshLambertMaterial(color=0x40ec00,
                                reflectivity=0.8)

    pose_z = t3d.transform_from_pos(jnp.array([0.0, 0.0, size/2.0]))
    objz = g.Box(np.array([size/10.0, size/10.0, size]))
    matz = g.MeshLambertMaterial(color=0x0b5cfc,
                                reflectivity=0.8)

    VISUALIZER[channel]["x"].set_object(objx, matx)
    VISUALIZER[channel]["x"].set_transform(np.array(pose @ pose_x, dtype=np.float64))
    VISUALIZER[channel]["y"].set_object(objy, maty)
    VISUALIZER[channel]["y"].set_transform(np.array(pose @ pose_y, dtype=np.float64))
    VISUALIZER[channel]["z"].set_object(objz, matz)
    VISUALIZER[channel]["z"].set_transform(np.array(pose @ pose_z, dtype=np.float64))
