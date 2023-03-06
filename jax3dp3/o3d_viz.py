import open3d as o3d
import numpy as np
import jax3dp3 as j

class O3DVis(object):
    def __init__(self, intrinsics):
        self.render = o3d.visualization.rendering.OffscreenRenderer(intrinsics.width, intrinsics.height)
        self.render.scene.set_background(np.array([1.0, 1.0, 1.0, 1.0]))

        self.counter = 0

    def set_camera(self, intrinsics, camera_pose):
        intr = o3d.camera.PinholeCameraIntrinsic(
            intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy
        )
        self.render.setup_camera(intr, np.linalg.inv(np.eye(4)))

        # Look at the origin from the front (along the -Z direction, into the screen), with Y as Up.
        center = np.array(camera_pose[:3,3]) + np.array(camera_pose[:3,2])  # look_at target
        eye = np.array(camera_pose[:3,3])  # camera position
        up = -np.array(camera_pose[:3,1])
        self.render.scene.camera.look_at(center, eye, up)

    def make_bounding_box(self, dims, pose, color=None, update=True):
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

        mtl = o3d.visualization.rendering.MaterialRecord()  # or MaterialRecord(), for later versions of Open3D
        mtl.shader = "defaultUnlit"
        self.render.scene.add_geometry(f"{self.counter}", line_set, mtl)
        self.counter+=1
        return line_set

    def make_cloud(self, cloud, color=None, update=True):
        if color is None:
            color = j.BLUE
        
        colors = np.tile(color, (cloud.shape[0],1))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        mtl = o3d.visualization.rendering.MaterialRecord()  # or MaterialRecord(), for later versions of Open3D
        mtl.shader = "defaultUnlit"

        self.render.scene.add_geometry(f"{self.counter}", pcd, mtl)
        self.counter+=1
        return pcd

    def make_mesh(self, filename, pose):
        mesh = o3d.io.read_triangle_model(filename)
        mesh.meshes[0].mesh.transform(pose)
        self.render.scene.add_model(f"{self.counter}", mesh)
        self.counter+=1


    def capture_image(self):
        img = np.array(self.render.render_to_image())
        return img
