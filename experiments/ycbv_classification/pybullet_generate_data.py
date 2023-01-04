import os
import numpy as np
import pybullet as p
import jax
import jax3dp3
import jax3dp3.mesh
import trimesh
import jax.numpy as jnp
import jax3dp3.pybullet
import jax3dp3.transforms_3d as t3d


h, w, fx,fy, cx,cy = (
    480,
    640,
    500.0,500.0,
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
idx_1 = 0
idx_2 = 1
cracker_box_path = os.path.join(jax3dp3.utils.get_assets_dir(), "models/{}/textured_simple.obj".format(model_names[idx_1]))
cracker_box_mesh = trimesh.load(cracker_box_path)
cracker_box_mesh_centered = jax3dp3.mesh.center_mesh(cracker_box_mesh)
cracker_box_centered_path = "/tmp/obj1/obj.obj"
os.makedirs(os.path.dirname(cracker_box_centered_path),exist_ok=True)
cracker_box_mesh_centered.export(cracker_box_centered_path)

sugar_box_path = os.path.join(jax3dp3.utils.get_assets_dir(), "models/{}/textured_simple.obj".format(model_names[idx_2]))
sugar_box_mesh = trimesh.load(sugar_box_path)
sugar_box_mesh_centered = jax3dp3.mesh.center_mesh(sugar_box_mesh)
sugar_box_centered_path = "/tmp/obj2/obj.obj"
os.makedirs(os.path.dirname(sugar_box_centered_path),exist_ok=True)
sugar_box_mesh_centered.export(sugar_box_centered_path)

table_mesh = jax3dp3.mesh.center_mesh(jax3dp3.mesh.make_table_mesh(
    1.0,
    2.0,
    0.8,
    0.01,
    0.01
))
table_mesh_path  = "/tmp/table/table.obj"
os.makedirs(os.path.dirname(table_mesh_path),exist_ok=True)
table_mesh.export(table_mesh_path)

table, table_dims = jax3dp3.pybullet.add_mesh(table_mesh_path)
object, obj_dims = jax3dp3.pybullet.add_mesh(cracker_box_centered_path)
object2, obj_dims2 = jax3dp3.pybullet.add_mesh(sugar_box_centered_path)


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

get_poses_jit = jax.jit(jax3dp3.scene_graph.absolute_poses_from_scene_graph)
poses = get_poses_jit(
    absolute_poses, box_dims, edges, contact_params, face_params
)
for (obj, pose) in zip(all_pybullet_objects, poses):
    jax3dp3.pybullet.set_pose_wrapped(obj, pose)


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



head_joint_names = ["head_pan_joint", "head_tilt_joint"]
head_joints = jax3dp3.pybullet.joints_from_names(robot, head_joint_names)


all_data = [] 
start_position = jnp.array([-1.0, 0.0, 0.0])
vel = jnp.array([-0.02, 0.0, 0.0])
num_timesteps = 50
for t in range(num_timesteps):
    robot_pose = t3d.transform_from_pos(start_position + vel * t)
    jax3dp3.pybullet.set_pose_wrapped(robot, robot_pose)
    conf = jax3dp3.pybullet.inverse_visibility(robot, jax3dp3.pybullet.get_pose_wrapped(object)[:3,3], verbose=True, tolerance=0.001)
    jax3dp3.pybullet.set_joint_positions(robot, head_joints, conf)

    cam_pose = jax3dp3.pybullet.get_link_pose_wrapped(
        robot, jax3dp3.pybullet.link_from_name(robot,  "head_mount_kinect_rgb_optical_frame")
    )
    rgb, depth, segmentation = jax3dp3.pybullet.capture_image(
        cam_pose,
        h, w, fx,fy, cx,cy , near, far
    )
    all_data.append((rgb,depth,segmentation, cam_pose))

jax3dp3.viz.make_gif_from_pil_images([
    jax3dp3.viz.get_rgba_image(data[0], 255.0)
    for data in all_data
], "rgb.gif")

np.savez("data.npz", rgb=[data[0] for data in all_data], 
         depth=[data[1] for data in all_data],
         segmentation=[data[2] for data in all_data],
         params=(h,w,fx,fy,cx,cy,near,far),
         cam_pose=[data[3] for data in all_data],
         table_pose=poses[0],
         table_dims=box_dims[0],
         all_object_poses=poses
)






from IPython import embed; embed()
