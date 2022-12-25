import trimesh
import jax3dp3.bbox
import numpy as np
import sys

filename = sys.argv[1]

scaling = 1000.0
swaps = np.array([1.0, 1.0, 1.0])

mesh = trimesh.load(filename)
bbox_dims, bbox_pose = jax3dp3.bbox.axis_aligned_bounding_box(mesh.vertices)
print(- bbox_pose[:3,3])
mesh.vertices = mesh.vertices - bbox_pose[:3,3]
mesh.vertices = mesh.vertices * scaling * swaps
bbox_dims, bbox_pose = jax3dp3.bbox.axis_aligned_bounding_box(mesh.vertices)
print(bbox_dims)
print(bbox_pose[:3,3])
mesh.export(filename)
