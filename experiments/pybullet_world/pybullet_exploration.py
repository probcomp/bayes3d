import numpy as np
import jax
import pybullet as p
import jax.numpy as jnp
import jax3dp3.utils
import os
import jax3dp3.transforms_3d as t3d
import jax3dp3.bbox
import jax3dp3.pybullet_utils as jpb
import jax3dp3.camera
import jax3dp3.viz
from jax3dp3.scene_graph import get_poses


height, width, fx,fy, cx,cy = (
    480,
    640,
    1000.0,1000.0,
    320.0,240.0
)
near,far = 0.1, 20.0

p.connect(p.GUI)
# p.connect(p.DIRECT)

ycb_path1 = os.path.join(jax3dp3.utils.get_assets_dir(), "models/003_cracker_box/textured_simple.obj")
ycb_path2 = os.path.join(jax3dp3.utils.get_assets_dir(), "models/004_sugar_box/textured_simple.obj")

table, table_dims = jpb.create_table(
    1.0,
    2.0,
    0.8,
    0.01,
    0.01
)
object, obj_dims = jpb.create_obj_centered(ycb_path1)
object2, obj_dims2 = jpb.create_obj_centered(ycb_path2)


all_pybullet_objects = [
    table,
    object,
    object2
]

box_dims = jnp.array([
    table_dims,
    obj_dims,
    obj_dims2
])


absolute_poses = jnp.array([
    t3d.transform_from_pos(jnp.array([0.0, 0.0, table_dims[2]/2.0])),
    jnp.eye(4),
    jnp.eye(4)
])

edges = jnp.array([
    [-1,0],
    [0,1],
    [0,2],
])

contact_params = jnp.array(
    [
        [0.0, 0.0, jnp.pi/4],
        [0.0, 0.0, jnp.pi/4],
        [0.1, 0.1, -jnp.pi/4],
    ]
)

face_params = jnp.array(
    [
        [2,3],
        [2,3],
        [2,3],
    ]
)

get_poses_jit = jax.jit(get_poses)
poses = get_poses_jit(
    absolute_poses, box_dims, edges, contact_params, face_params
)
for (obj, pose) in zip(all_pybullet_objects, poses):
    jpb.set_pose_wrapped(obj, pose)


# cam_pose = t3d.transform_from_rot_and_pos(
#     jnp.array([
#         [0.0, 0.0, -1.0],
#         [1.0, 0.0, 0.0],
#         [0.0, -1.0, 0.0],
#     ]),
#     jnp.array([2.0, 0.0, 1.0])
# )

pr2_urdf = os.path.join(jax3dp3.utils.get_assets_dir(), "robots/pr2/pr2.urdf")
robot = p.loadURDF(pr2_urdf, useFixedBase=True)


new_robot_pose = t3d.transform_from_pos(jnp.array([-2.0, 1.0, 0.0]))
jpb.set_pose_wrapped(robot, new_robot_pose)

head_joint_names = ["head_pan_joint", "head_tilt_joint"]
head_joints = jpb.joints_from_names(robot, head_joint_names)
conf = jpb.inverse_visibility(robot, jpb.get_pose_wrapped(object)[:3,3], verbose=True, tolerance=0.001)
jpb.set_joint_positions(robot, head_joints, conf)


print(jpb.get_joint_positions(robot, head_joints))


cam_pose = jpb.get_link_pose_wrapped(robot, jpb.link_from_name(robot,  "head_mount_kinect_rgb_optical_frame"))
rgb, depth, segmentation = jpb.capture_image(
    cam_pose,
    height, width, fx,fy, cx,cy , near, far
)
jax3dp3.viz.save_rgba_image(rgb, 255.0, "rgb.png")
jax3dp3.viz.save_depth_image(depth, "depth.png", max=far)
jax3dp3.viz.save_depth_image(segmentation, "seg.png", min=-1.0,max=4.0)

jpb.set_arm_conf(robot, "left", jpb.COMPACT_LEFT_ARM)



from IPython import embed; embed()
