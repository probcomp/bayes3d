import cv2
import os
import functools
import time
import trimesh
import numpy as np
import jax.numpy as jnp
import jax

import bayes3d
import bayes3d.transforms_3d as t3d
import matplotlib.pyplot as plt


## choose a test image  
# scene_id = '49'     # 48 ... 59
# img_ids = ['1', '526', '762', '1575', '2196']

scene_id = '48'     # 48 ... 59
img_ids = ['1', '1094', '1576','1622','2040', '2160']

## choose gt object to learn
obj_number = 4

def get_obj_cloud_from_img(img_id, obj_number):

    # process image
    test_img = bayes3d.ycb_loader.get_test_img(scene_id, img_id, os.environ["YCB_DIR"])
    depth_data = test_img.get_depth_image()

    segmentation_data = test_img.get_segmentation_image() 
    gt_ycb_idx = test_img.get_gt_indices()[obj_number]
    print("GT ycb idx=", gt_ycb_idx)

    ## retrieve poses
    cam_pose = test_img.get_camera_pose()
    table_pose = jnp.eye(4)  # xy plane 

    ## setup intrinsics
    orig_h, orig_w = test_img.get_image_dims()
    fx, fy, cx, cy = test_img.get_camera_intrinsics()
    # h, w, fx, fy, cx, cy  = jax3dp3.camera.scale_camera_parameters(orig_h,orig_w,fx,fy,cx,cy, 0.25)
    h, w = orig_h, orig_w
    print("intrinsics:", h, w, fx, fy, cx, cy)

    gt_img = t3d.depth_to_point_cloud_image(depth_data * (segmentation_data == obj_number), fx,fy,cx,cy)

    gt_image_single_object_cloud = t3d.point_cloud_image_to_points(gt_img)
    gt_points_in_table_frame = t3d.apply_transform(
        t3d.apply_transform(gt_image_single_object_cloud, cam_pose), 
        jnp.linalg.inv(table_pose)
    )
    # point_seg = jax3dp3.utils.segment_point_cloud(gt_image_single_object_cloud, 2.0)
    # gt_points_in_table_frame = gt_points_in_table_frame[point_seg == jax3dp3.utils.get_largest_cluster_id_from_segmentation(point_seg)]
    # print(f"gt_points_in_table_frame with {len(gt_points_in_table_frame)} points")

    return gt_points_in_table_frame




all_cloud_list = []
for img_id in img_ids:
    cloud_in_table_frame = get_obj_cloud_from_img(img_id, obj_number)
    
    print(f"appending cloud with {cloud_in_table_frame.shape[0]} points")
    all_cloud_list.append(cloud_in_table_frame)


all_cloud = jnp.vstack(all_cloud_list)
print(f"total object cloud in table frame, {all_cloud.shape[0]} points")

plt.figure()
for cloud in all_cloud_list:
    plt.scatter(cloud[:,0], cloud[:,1])

plt.savefig("cloud.png")


bayes3d.meshcat.setup_visualizer()

bayes3d.meshcat.show_cloud("cloud", all_cloud / 100.0)

from IPython import embed; embed()


import open3d as o3d
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(all_cloud)
pcd.estimate_normals()
radius = 1.0 #???
mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
    pcd, o3d.utility.DoubleVector([radius, radius*2.0]))
tri_mesh = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles),
            vertex_normals=np.asarray(mesh.vertex_normals))
bayes3d.meshcat.show_trimesh("obj", bayes3d.mesh.scale_mesh(tri_mesh, 0.1))
# jax3dp3.meshcat.clear()
bayes3d.meshcat.VISUALIZER.delete()



bbox = bayes3d.utils.aabb(cloud)

from IPython import embed; embed()
