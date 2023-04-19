import jax3dp3.nvdiffrast.common as dr
import torch
import jax3dp3.camera
import jax3dp3 as j
import jax3dp3.transforms_3d as t3d
import trimesh
import jax.numpy as jnp
import jax
import numpy as np
import jax.dlpack


class RGBD(object):
    def __init__(self, rgb, depth, camera_pose, intrinsics, segmentation=None):
        self.intrinsics = intrinsics
        self.rgb = rgb
        self.depth = depth
        self.camera_pose = camera_pose
        self.segmentation = segmentation

    def construct_from_camera_image(camera_image, near=0.001, far=5.0):
        depth = np.array(camera_image.depthPixels)
        rgb = np.array(camera_image.rgbPixels)
        camera_pose = t3d.pybullet_pose_to_transform(camera_image.camera_pose)
        K = camera_image.camera_matrix
        fx, fy, cx, cy = K[0,0],K[1,1],K[0,2],K[1,2]
        h,w = depth.shape
        near = 0.001
        return RGBD(rgb, depth, camera_pose, j.Intrinsics(h,w,fx,fy,cx,cy,near,far))

    def construct_from_aidan_dict(d, near=0.001, far=5.0):
        depth = np.array(d["depth"] / 1000.0) 
        camera_pose = t3d.pybullet_pose_to_transform(d["extrinsics"])
        rgb = np.array(d["rgb"])
        K = d["intrinsics"][0]
        fx, fy, cx, cy = K[0,0],K[1,1],K[0,2],K[1,2]
        h,w = depth.shape
        observation = RGBD(rgb, depth, camera_pose, j.Intrinsics(h,w,fx,fy,cx,cy,near,far))
        return observation

    def construct_from_step_metadata(step_metadata, intrinsics=None):
        if intrinsics is None:
            width, height = step_metadata.camera_aspect_ratio
            aspect_ratio = width / height
            cx, cy = width / 2.0, height / 2.0
            fov_y = np.deg2rad(step_metadata.camera_field_of_view)
            fov_x = 2 * np.arctan(aspect_ratio * np.tan(fov_y / 2.0))
            fx = cx / np.tan(fov_x / 2.0)
            fy = cy / np.tan(fov_y / 2.0)
            clipping_near, clipping_far = step_metadata.camera_clipping_planes
            intrinsics = j.Intrinsics(
                height,width, fx,fy,cx,cy,clipping_near,clipping_far
            )

        rgb = np.array(list(step_metadata.image_list)[-1])
        depth = np.array(list(step_metadata.depth_map_list)[-1])
        seg = np.array(list(step_metadata.object_mask_list)[-1])
        colors, seg_final_flat = np.unique(seg.reshape(-1,3), axis=0, return_inverse=True)
        seg_final = seg_final_flat.reshape(seg.shape[:2])
        observation = RGBD(rgb, depth, jnp.eye(4), intrinsics, seg_final)
        return observation


class Renderer(object):
    def __init__(self, intrinsics, num_layers=1024):
        self.intrinsics = intrinsics
        self.renderer_env = dr.RasterizeGLContext(intrinsics.height, intrinsics.width, output_db=False)
        self.proj_list = list(jax3dp3.camera.open_gl_projection_matrix(
            intrinsics.height, intrinsics.width, intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy, intrinsics.near, intrinsics.far
        ).reshape(-1))
        
        dr._get_plugin(gl=True).setup(
            self.renderer_env.cpp_wrapper,
            intrinsics.height, intrinsics.width, num_layers
        )

        self.meshes =[]
        self.mesh_names =[]
        self.model_box_dims = jnp.zeros((0,3))

    def add_mesh_from_file(self, mesh_filename, mesh_name=None, scaling_factor=1.0, force=None):
        mesh = trimesh.load(mesh_filename, force=force)
        self.add_mesh(mesh, mesh_name=mesh_name, scaling_factor=scaling_factor)

    def add_mesh(self, mesh, mesh_name=None, scaling_factor=1.0):
        
        if mesh_name is None:
            mesh_name = f"object_{len(self.meshes)}"
        
        mesh.vertices = mesh.vertices * scaling_factor
        self.meshes.append(mesh)
        self.mesh_names.append(mesh_name)

        self.model_box_dims = jnp.vstack(
            [
                self.model_box_dims,
                jax3dp3.utils.aabb(mesh.vertices)[0]
            ]
        )

        vertices = np.array(mesh.vertices)
        vertices = np.concatenate([vertices, np.ones((*vertices.shape[:-1],1))],axis=-1)
        triangles = np.array(mesh.faces)
        dr._get_plugin(gl=True).load_vertices_fwd(
            self.renderer_env.cpp_wrapper, torch.tensor(vertices.astype("f"), device='cuda'),
            torch.tensor(triangles.astype(np.int32), device='cuda'),
        )

    def render_to_torch(self, poses, idx, on_object=0):
        poses_torch = torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(poses))
        images_torch = dr._get_plugin(gl=True).rasterize_fwd_gl(self.renderer_env.cpp_wrapper, poses_torch, self.proj_list, idx, on_object)
        return images_torch

    def render_single_object(self, pose, idx):
        images_torch = self.render_to_torch(pose[None, None, :, :], [idx])
        return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(images_torch[0]))

    def render_parallel(self, poses, idx):
        images_torch = self.render_to_torch(poses[None, :, :, :], [idx])
        return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(images_torch))

    def render_multiobject(self, poses, indices):
        images_torch = self.render_to_torch(poses[:, None, :, :], indices)
        return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(images_torch[0]))

    def render_multiobject_parallel(self, poses, indices, on_object=0):
        images_torch = self.render_to_torch(poses, indices, on_object=on_object)
        return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(images_torch))

def render_point_cloud(point_cloud, intrinsics, pixel_smudge=0):
    transformed_cloud = point_cloud
    point_cloud = jnp.vstack([jnp.zeros((1, 3)), transformed_cloud])
    pixels = project_cloud_to_pixels(point_cloud, intrinsics)
    x, y = jnp.meshgrid(jnp.arange(intrinsics.width), jnp.arange(intrinsics.height))
    matches = (jnp.abs(x[:, :, None] - pixels[:, 0]) <= pixel_smudge) & (jnp.abs(y[:, :, None] - pixels[:, 1]) <= pixel_smudge)
    matches = matches * (intrinsics.far * 2.0 - point_cloud[:,-1][None, None, :])
    a = jnp.argmax(matches, axis=-1)    
    return point_cloud[a]

def render_point_cloud_batched(point_cloud, intrinsics, NUM_PER, pixel_smudge=0):
    all_images = []
    num_iters = jnp.ceil(point_cloud.shape[0] / NUM_PER).astype(jnp.int32)
    for i in tqdm(range(num_iters)):
        img = j.render_point_cloud(point_cloud[i*NUM_PER:i*NUM_PER+NUM_PER], intrinsics)
        img = img.at[img[:,:,2] < intrinsics.near].set(intrinsics.far)
        all_images.append(img)
    all_images_stack = jnp.stack(all_images,axis=-2)
    best = all_images_stack[:,:,:,2].argmin(-1)
    img = all_images_stack[
        np.arange(intrinsics.height)[:, None],
        np.arange(intrinsics.width)[None, :],
        best,
        :
    ]
    return img

def project_cloud_to_pixels(point_cloud, intrinsics):
    point_cloud_normalized = point_cloud / point_cloud[:, 2].reshape(-1, 1)
    temp1 = point_cloud_normalized[:, :2] * jnp.array([intrinsics.fx,intrinsics.fy])
    temp2 = temp1 + jnp.array([intrinsics.cx, intrinsics.cy])
    pixels = jnp.round(temp2) 
    return pixels

def get_masked_and_complement_image(depth_image, segmentation_image, segmentation_id, intrinsics):
    mask =  (segmentation_image == segmentation_id)
    masked_image = depth_image * mask + intrinsics.far * (1.0 - mask)
    complement_image = depth_image * (1.0 - mask) + intrinsics.far * mask
    return masked_image, complement_image


def splice_image_parallel(rendered_object_image, obs_image_complement):
    keep_masks = jnp.logical_or(
        (rendered_object_image[:,:,:,2] <= obs_image_complement[None, :,:, 2]) * 
        rendered_object_image[:,:,:,2] > 0.0
        ,
        (obs_image_complement[:,:,2] == 0)[None, ...]
    )[...,None]
    rendered_images = keep_masks * rendered_object_image + (1.0 - keep_masks) * obs_image_complement
    return rendered_images
