import meshcat
import numpy as np
import meshcat.geometry as g
VISUALIZER = None
from matplotlib.colors import rgb2hex

def setup_visualizer():
    global VISUALIZER
    VISUALIZER = meshcat.Visualizer()
    set_background_color([1, 1, 1])

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
    point_cloud = np.transpose(np.array(point_cloud))
    if color is None:
        color = np.zeros_like(point_cloud)
    color = np.array(color)
    obj = g.PointCloud(point_cloud, color, size=size)
    VISUALIZER[channel].set_object(obj)

def show_trimesh(channel, mesh, color=None, wireframe=False):
    global VISUALIZER
    if color is None:
        color = [1, 0, 0]
    material = g.MeshLambertMaterial(color=int(rgb2hex(color)[1:],16), wireframe=wireframe)
    obj = g.TriangularMeshGeometry(mesh.vertices, mesh.faces)
    VISUALIZER[channel].set_object(obj, material)
