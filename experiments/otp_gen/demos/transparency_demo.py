import open3d as o3d
import numpy as np

# Create a transparent box as a see-through wall
mesh1 = o3d.geometry.TriangleMesh.create_box(width=2.0, height=2.0, depth=0.1)
mesh1.compute_vertex_normals()
mesh1.translate((0, 0, 1))  # Move the box forward

mat1 = o3d.visualization.rendering.MaterialRecord()
mat1.shader = "defaultLitTransparency"
mat1.base_color = np.array([1, 0, 0, 0.75])  # Red color with 25% transparency

# Create an opaque unit sphere
mesh2 = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
mesh2.compute_vertex_normals()

mat2 = o3d.visualization.rendering.MaterialRecord()
mat2.shader = "defaultLit"
mat2.base_color = np.array([0, 0, 1, 1])  # Blue color with no transparency

# Visualize the box and sphere
o3d.visualization.draw([
    {'name': 'transparent_box', 'geometry': mesh1, 'material': mat1},
    {'name': 'opaque_sphere', 'geometry': mesh2, 'material': mat2}
])
