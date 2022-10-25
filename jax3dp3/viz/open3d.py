import open3d as o3d

global vis

def open_window():
    global vis
    vis = o3d.visualization.Visualizer()
    vis.create_window()


