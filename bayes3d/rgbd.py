import bayes3d.camera
import bayes3d as j
import bayes3d as b
import bayes3d.transforms_3d as t3d
import numpy as npe
import jax.numpy as jnp
import numpy as np

class RGBD(object):
    def __init__(self, rgb, depth, camera_pose, intrinsics, segmentation=None):
        """RGBD Image
        
        Args:
            rgb (np.array): RGB image
            depth (np.array): Depth image
            camera_pose (np.array): Camera pose. 4x4 matrix
            intrinsics (b.camera.Intrinsics): Camera intrinsics
            segmentation (np.array): Segmentation image
        """
        self.rgb = rgb
        self.depth = depth
        self.camera_pose = camera_pose
        self.intrinsics = intrinsics
        self.segmentation  = segmentation

    def construct_from_camera_image(camera_image, near=0.001, far=5.0):
        """Construct RGBD image from CameraImage
        
        Args:
            camera_image (CameraImage): CameraImage object
        Returns:
            RGBD: RGBD image
        """
        depth = np.array(camera_image.depthPixels)
        rgb = np.array(camera_image.rgbPixels)
        camera_pose = t3d.pybullet_pose_to_transform(camera_image.camera_pose)
        K = camera_image.camera_matrix
        fx, fy, cx, cy = K[0,0],K[1,1],K[0,2],K[1,2]
        h,w = depth.shape
        near = 0.001
        return RGBD(rgb, depth, camera_pose, j.Intrinsics(h,w,fx,fy,cx,cy,near,far))

    def construct_from_aidan_dict(d, near=0.001, far=5.0):
        """Construct RGBD image from Aidan's dictionary
        
        Args:
            d (dict): Dictionary containing rgb, depth, extrinsics, intrinsics
        Returns:
            RGBD: RGBD image
        """
        depth = np.array(d["depth"] / 1000.0) 
        camera_pose = t3d.pybullet_pose_to_transform(d["extrinsics"])
        rgb = np.array(d["rgb"])
        K = d["intrinsics"][0]
        fx, fy, cx, cy = K[0,0],K[1,1],K[0,2],K[1,2]
        h,w = depth.shape
        observation = RGBD(rgb, depth, camera_pose, j.Intrinsics(h,w,fx,fy,cx,cy,near,far))
        return observation

    def construct_from_step_metadata(step_metadata, intrinsics=None):
        """Construct RGBD image from StepMetadata.

        Args:
            step_metadata (StepMetadata): StepMetadata object
        Returns:
            RGBD: RGBD image
        """
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

    def scale_rgbd(self, scaling_factor):
        rgb = b.utils.scale(self.rgb, scaling_factor)
        depth= b.utils.scale(self.depth, scaling_factor)
        intrinsics = b.camera.scale_camera_parameters(self.intrinsics, scaling_factor)
        return RGBD(rgb, depth, self.camera_pose, intrinsics, self.segmentation)