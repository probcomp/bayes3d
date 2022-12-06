import open3d as o3d
import numpy as np
vis = o3d.visualization.Visualizer()
vis.create_window()
widths = [1.0, 2.0, 3.0]
box_geom = o3d.geometry.TriangleMesh.create_box(
    width=widths[0],
    height=widths[1],
    depth=widths[2],
)

pose = np.eye(4)
pose[:3,3] = np.array([2.0, 3.0, 5.0])

box_geom.transform(pose)
vis.add_geometry(box_geom)

vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0))

vis.run()