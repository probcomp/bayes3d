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
import jax3dp3.mesh
import trimesh


h, w, fx,fy, cx,cy = (
    480,
    640,
    1000.0,1000.0,
    320.0,240.0
)
near,far = 0.01, 50.0

# h, w = 120, 160
# fx,fy = 200.0, 200.0
# cx,cy = 80.0, 60.0
# near=0.01
# far=50.0
# max_depth=2.0


p.connect(p.GUI)
# p.connect(p.DIRECT)

model_dir = os.path.join(jax3dp3.utils.get_assets_dir(),"models")
model_names = os.listdir(model_dir)
idx_1 = 9
idx_2 = 10
cracker_box_path = os.path.join(jax3dp3.utils.get_assets_dir(), "models/{}/textured_simple.obj".format(model_names[idx_1]))
cracker_box_mesh = trimesh.load(cracker_box_path)
cracker_box_mesh_centered = jax3dp3.mesh.center_mesh(cracker_box_mesh)
cracker_box_centered_path = "/tmp/cracker_box/cracker_box.obj"
os.makedirs(os.path.dirname(cracker_box_centered_path),exist_ok=True)
cracker_box_mesh_centered.export(cracker_box_centered_path)

sugar_box_path = os.path.join(jax3dp3.utils.get_assets_dir(), "models/{}/textured_simple.obj".format(model_names[idx_2]))
sugar_box_mesh = trimesh.load(sugar_box_path)
sugar_box_mesh_centered = jax3dp3.mesh.center_mesh(sugar_box_mesh)
sugar_box_centered_path = "/tmp/sugar_box/sugar_box.obj"
os.makedirs(os.path.dirname(sugar_box_centered_path),exist_ok=True)
sugar_box_mesh_centered.export(sugar_box_centered_path)

table_mesh = jax3dp3.mesh.center_mesh(jax3dp3.mesh.make_table_mesh(
    1.0,
    2.0,
    0.8,
    0.01,
    0.01
))
table_mesh_path  = "table.obj"
table_mesh.export(table_mesh_path)

table, table_dims = jpb.add_mesh(table_mesh_path)
object, obj_dims = jpb.add_mesh(cracker_box_centered_path)
object2, obj_dims2 = jpb.add_mesh(sugar_box_centered_path)


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
        [0.2, 0.2, -jnp.pi/4],
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


new_robot_pose = t3d.transform_from_pos(jnp.array([-1.0, 1.0, 0.0]))
jpb.set_pose_wrapped(robot, new_robot_pose)

head_joint_names = ["head_pan_joint", "head_tilt_joint"]
head_joints = jpb.joints_from_names(robot, head_joint_names)
conf = jpb.inverse_visibility(robot, jpb.get_pose_wrapped(object)[:3,3], verbose=True, tolerance=0.001)
jpb.set_joint_positions(robot, head_joints, conf)


print(jpb.get_joint_positions(robot, head_joints))


cam_pose = jpb.get_link_pose_wrapped(robot, jpb.link_from_name(robot,  "head_mount_kinect_rgb_optical_frame"))
rgb, depth, segmentation = jpb.capture_image(
    cam_pose,
    h, w, fx,fy, cx,cy , near, far
)
jax3dp3.viz.save_rgba_image(rgb, 255.0, "rgb.png")
jax3dp3.viz.save_depth_image(depth, "depth.png", max=far)
jax3dp3.viz.save_depth_image(segmentation, "seg.png", min=-1.0,max=4.0)

jpb.set_arm_conf(robot, "left", jpb.COMPACT_LEFT_ARM)

np.savez("data.npz", rgb=rgb, depth=depth, segmentation=segmentation, params=(h,w,fx,fy,cx,cy,near,far), cam_pose=cam_pose, table_pose=poses[0], table_dims=box_dims[0])



from IPython import embed; embed()
