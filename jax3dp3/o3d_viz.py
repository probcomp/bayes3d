import open3d as o3d
import numpy as np
import jax3dp3 as j

def make_camera(intrinsics, pose):
    cam = o3d.camera.PinholeCameraParameters()
    intr = o3d.camera.PinholeCameraIntrinsic(
        intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy
    )
    cam.intrinsic = intr
    cam.extrinsic = np.linalg.inv(pose)
    return cam

def setup(intrinsics):
    global VISUALIZER
    VISUALIZER = o3d.visualization.Visualizer()
    VISUALIZER.create_window(width=intrinsics.width, height=intrinsics.height)

    mesh = o3d.geometry.TriangleMesh.create_box()
    VISUALIZER.add_geometry(mesh)


    cam = make_camera(intrinsics, np.eye(4))

    view = VISUALIZER.get_view_control()
    view.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)
    VISUALIZER.remove_geometry(mesh)
    VISUALIZER.poll_events()
    VISUALIZER.update_renderer()

#Updates visualizaiton to most recent events 
def sync():
    global VISUALIZER
    VISUALIZER.poll_events()
    VISUALIZER.update_renderer()

#Activates the current visualizer window
def run():
    global VISUALIZER
    VISUALIZER.run()
    VISUALIZER.destroy_window()
    VISUALIZER.close()

def destroy():
    global VISUALIZER
    VISUALIZER.destroy_window()
    VISUALIZER.close()

def add(geometry, update=True):
    global VISUALIZER
    VISUALIZER.add_geometry(geometry)
    if update:
        sync()

def clear():
    global VISUALIZER
    VISUALIZER.clear_geometries()
    sync()
 

#Returns an Open3D Image of the current visualizaiton
def capture_image():
    global VISUALIZER
    img_buf = VISUALIZER.capture_screen_float_buffer()
    return np.array(img_buf) * 255.0

def set_camera(intrinsics, pose):
    global VISUALIZER
    cam = make_camera(intrinsics, pose)
    view = VISUALIZER.get_view_control()
    view.convert_from_pinhole_camera_parameters(cam, allow_arbitrary=True)
    VISUALIZER.poll_events()
    VISUALIZER.update_renderer()

def make_bounding_box(dims, pose, line_set, color=None, update=True):
    if line_set is None:
        line_set = o3d.geometry.LineSet()

    if color is None:
        color = j.RED   
    
    points = np.zeros((9,3))
    points[0, :] = np.array([dims[0]/2, -dims[1]/2, dims[2]/2]  )
    points[1, :] = np.array([-dims[0]/2, -dims[1]/2, dims[2]/2])
    points[2, :] = np.array([-dims[0]/2, dims[1]/2, dims[2]/2])
    points[3, :] = np.array([dims[0]/2, dims[1]/2, dims[2]/2])
    points[4, :] = np.array([dims[0]/2, -dims[1]/2, -dims[2]/2])
    points[5, :] = np.array([-dims[0]/2, -dims[1]/2, -dims[2]/2])
    points[6, :] = np.array([-dims[0]/2, dims[1]/2, -dims[2]/2])
    points[7, :] = np.array([dims[0]/2, dims[1]/2, -dims[2]/2])
    points[8, :] = np.array([0.0, 0.0, 0.0])

    new_points = j.t3d.apply_transform(points, pose)

    lines = np.array([ 
        [1,2],
        [2,3],
        [3,4],
        [4,1],
        [5,6],
        [6,7],
        [7,8],
        [8,5],
        [1,5],
        [2,6],
        [3,7],
        [4,8] 
    ]) - 1

    line_set.points =  o3d.utility.Vector3dVector(new_points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.paint_uniform_color(color)
    add(line_set, update=update)
    return line_set


def make_cloud(cloud, pcd, color=None, update=True):
    if color is None:
        color = j.BLUE
    
    colors = np.tile(color, (cloud.shape[0],1))

    if pcd is None:
        pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(cloud)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    add(pcd, update=update)
    return pcd

def make_mesh_from_file(filename, mesh, pose=None, color=None, update=True):
    m = o3d.io.read_triangle_mesh(filename, True)
    if pose is not None:
        m.transform(np.array(pose))
    add(m, update=update)
    return m