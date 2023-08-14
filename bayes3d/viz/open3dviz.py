import open3d as o3d
import numpy as np
import bayes3d as j
import bayes3d as b
import jax.numpy as jnp

def trimesh_to_o3d_triangle_mesh(trimesh_mesh):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices =  o3d.utility.Vector3dVector(trimesh_mesh.vertices)
    mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
    mesh.triangle_normals = o3d.utility.Vector3dVector(np.array(trimesh_mesh.face_normals))
    return mesh

class Open3DVisualizer(object):
    def __init__(self, intrinsics):
        self.render = o3d.visualization.rendering.OffscreenRenderer(intrinsics.width, intrinsics.height)
        # self.set_background(np.array([0.0, 0.0, 0.0, 0.0]))
        self.render.scene.set_background(np.array([1.0, 1.0, 1.0, 1.0]))
        self.render.scene.set_lighting(self.render.scene.LightingProfile.NO_SHADOWS, (0, 0, 0))

        self.counter = 0

    def set_background(self, background):
        self.render.scene.set_background(background)


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
        
        if color.shape[0] != cloud.shape[0]:
            colors = np.tile(color, (cloud.shape[0],1))
        else:
            colors = color

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        mtl = o3d.visualization.rendering.MaterialRecord()  # or MaterialRecord(), for later versions of Open3D
        mtl.shader = "defaultUnlit"

        self.render.scene.add_geometry(f"{self.counter}", pcd, mtl)
        self.counter+=1
        return pcd

    def make_mesh_from_file(self, filename, pose, scaling_factor=1.0):
        mesh = o3d.io.read_triangle_model(filename)
        mesh.meshes[0].mesh.scale(scaling_factor, np.array([0.0, 0.0, 0.0]))
        mesh.meshes[0].mesh.transform(pose)
        self.render.scene.add_model(f"{self.counter}", mesh)
        self.counter+=1
    
    def clear(self):
        self.render.scene.clear_geometry()

    def make_trimesh(self, trimesh_mesh, pose, color):
        mesh = trimesh_to_o3d_triangle_mesh(trimesh_mesh)
        mesh.transform(pose)
        mtl = o3d.visualization.rendering.MaterialRecord()
        mtl.shader = 'defaultLitTransparency'
        mtl.base_color = color

        self.render.scene.add_geometry(f"{self.counter}", mesh, mtl)
        self.counter+=1

    def capture_image(self, intrinsics, camera_pose):
        self.render.scene.camera.set_projection(
            b.camera.K_from_intrinsics(intrinsics),
            intrinsics.near, intrinsics.far,
            intrinsics.width,
            intrinsics.height
        )
        # Look at the origin from the front (along the -Z direction, into the screen), with Y as Up.
        center = np.array(camera_pose[:3,3]) + np.array(camera_pose[:3,2])  # look_at target
        eye = np.array(camera_pose[:3,3])  # camera position
        up = -np.array(camera_pose[:3,1])
        self.render.scene.camera.look_at(center, eye, up)
        img = np.array(self.render.render_to_image())
        rgb = j.add_rgba_dimension(img)
        depth = jnp.array(self.render.render_to_depth_image(z_in_view_space=True))

        return j.RGBD(rgb, depth, camera_pose, intrinsics)

    def make_camera(self, intrinsics, pose, size):
        cx = intrinsics.cx
        cy = intrinsics.cy
        fx = intrinsics.fx
        fy = intrinsics.fy
        width = intrinsics.width
        height = intrinsics.height

        color = j.BLUE
        dist=size
        vertices = np.zeros((5, 3))
        vertices[0, :] = [0, 0, 0]
        vertices[1, :] = [(0-cx)*dist/fx, (0-cy)*dist/fy, dist]
        vertices[2, :] = [(width-cx)*dist/fx, (0-cy)*dist/fy, dist]
        vertices[3, :] = [(width-cx)*dist/fx, (height-cy)*dist/fy, dist]
        vertices[4, :] = [(0-cx)*dist/fx, (height-cy)*dist/fy, dist]
        new_points = j.t3d.apply_transform(vertices, pose)
        lines = np.array([ 
            [0,1],
            [0,2],
            [0,3],
            [0,4],
            [1,2],
            [2,3],
            [3,4],
            [4,1],
        ])

        line_set = o3d.geometry.LineSet()
        line_set.points =  o3d.utility.Vector3dVector(new_points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.paint_uniform_color(color)

        mtl = o3d.visualization.rendering.MaterialRecord()  # or MaterialRecord(), for later versions of Open3D
        mtl.base_color = [0.0, 0.0, 1.0, 1.0]  # RGBA
        mtl.shader = "defaultUnlit"

        self.render.scene.add_geometry(f"{self.counter}", line_set, mtl)
        self.counter+=1

