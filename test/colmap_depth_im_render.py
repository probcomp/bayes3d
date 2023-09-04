import bayes3d as b
import jax.numpy as jnp
import numpy as np
import jax
import os
# import matplotlib.pyplot as plt
import trimesh

class ColmapScene:

    # scale ambiguity resolution not required to generate depth image
    def __init__(self, camera_intrinsic_file, camera_extrinsic_file, scene_mesh_file):
        cam_instrinsics = self.load_camera_intrinsics(camera_intrinsic_file)
        cam_extrinsics = self.load_camera_extrinsics(camera_extrinsic_file)
        self.scene_mesh_file = scene_mesh_file
        self.mesh = self.load_scene_mesh(scene_mesh_file)

        # each camera pose has separate intrinsics
        self.cameras = cam_extrinsics
        for ind, c in enumerate(self.cameras):
            # cameras is a list of paired intrinsics and extrinsics
            if len(cam_instrinsics) == len(cam_extrinsics):
                c.update(cam_instrinsics[ind])
            # more poses than intrinsics, assign first set of intrinsics to all poses
            else:
                c.update(cam_instrinsics[0])

    def load_camera_extrinsics(self, colmap_extrinsics_file):
        # colmap_extrinsics_file = "ku_scene_vids_linear/frames1/dense/0/sparse/images.txt"
        with open(colmap_extrinsics_file, 'r') as file:
            extrinsics_array = []
            for line in file:
                # Split each line into a list of floats using spaces as the delimiter
                if line[0] == "#":
                    continue
                float_list = [float(item) for item in line.strip().split()[1:-1]]
                extrinsics_array.append(float_list)

        extrinsics_array = np.array(extrinsics_array[::2])
        extrinsics_array = extrinsics_array[extrinsics_array[:, -1].argsort()][:,:-1]
        camera_extrinsics_list=[]

        for extrinsics in extrinsics_array:
            camera_extrinsics = {}
            camera_extrinsics['quaternion'] = extrinsics[0:4]
            camera_extrinsics['translation'] = extrinsics[4:]
            camera_extrinsics_list.append(camera_extrinsics)

        return camera_extrinsics_list

    def load_camera_intrinsics(self, colmap_intrinsics_file):
        # colmap_intrinsics_file = "ku_scene_vids_linear/frames1/dense/0/sparse/cameras.txt"
        with open(colmap_intrinsics_file, 'r') as file:
            intrinsics_array = []
            for line in file:
                # Split each line into a list of floats using spaces as the delimiter
                if line[0] == "#":
                    continue
                string_list = [item for item in line.strip().split()]
                float_list = [float(s) for s in [string_list[0]] + string_list[2:]]
                intrinsics_array.append(float_list)

        intrinsics_array = np.array(intrinsics_array)
        intrinsics_array = intrinsics_array[intrinsics_array[:, 0].argsort()][:,1:]
        camera_intrinsics_list=[]

        for intrinsics in intrinsics_array:
            camera_intrinsics = {}
            camera_intrinsics['colmap_width_height'] = intrinsics[0:2]
            camera_intrinsics['colmap_focal_lengths'] = intrinsics[2:4]
            camera_intrinsics['colmap_cx_cy'] = intrinsics[4:]
            camera_intrinsics_list.append(camera_intrinsics)

        return camera_intrinsics_list


    def load_scene_mesh(self, scene_mesh_file):
        # scene_mesh_file = 'ku_scene_vids_linear/frames1/dense/meshed_delaunay.ply'
        mesh = trimesh.load(scene_mesh_file)
        return mesh


    def get_depth_render_args(self, camera_num=0, scaling_factor=1):

        renderer_intrinsics = b.Intrinsics(
            width=self.cameras[camera_num]['colmap_width_height'][0], height=self.cameras[camera_num]['colmap_width_height'][1],
            fx=self.cameras[camera_num]['colmap_focal_lengths'][0], fy=self.cameras[camera_num]['colmap_focal_lengths'][1],
            cx=self.cameras[camera_num]['colmap_cx_cy'][0], cy=self.cameras[camera_num]['colmap_cx_cy'][1],
            near=0.1, far=200 #near=rgbd.intrinsics.near, far=rgbd.intrinsics.far
        )

        renderer_intrinsics = b.camera.scale_camera_parameters(renderer_intrinsics, scaling_factor)

        #b.setup_renderer(renderer_intrinsics)
        #b.RENDERER.add_mesh_from_file(self.scene_mesh_file, center_mesh=False)
       
        rot = b.t3d.quaternion_to_rotation_matrix(self.cameras[camera_num]['quaternion'])
        transform = b.t3d.transform_from_rot_and_pos(rot, self.cameras[camera_num]['translation'])
        #im_out = b.RENDERER.render(transform[None,...],jnp.array([0]))
        #depth_im = im_out[:,:,2]
        return renderer_intrinsics, self.scene_mesh_file, transform
    
    def get_pose_relative_to_basis(self, camera_num=0):
        rot = b.t3d.quaternion_to_rotation_matrix(self.cameras[camera_num]['quaternion'])
        transform_0 = b.t3d.transform_from_rot_and_pos(rot, self.cameras[camera_num]['translation'])

        transforms_im0_static = []
        for i in range(len(self.cameras)):
            rot = b.t3d.quaternion_to_rotation_matrix(self.cameras[i]['quaternion'])
            transform = b.t3d.transform_from_rot_and_pos(rot, self.cameras[i]['translation'])
            transform_im0_static = jnp.linalg.inv(transform)@transform_0
            transforms_im0_static.append(transform_im0_static)
        return transforms_im0_static
    
    def save_depth_im(depth_im, filename = 'depth_im_colmap_single.png'):
        jnp.save(filename, depth_im)


