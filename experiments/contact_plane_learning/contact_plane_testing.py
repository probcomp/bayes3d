import os
import numpy as np
import pybullet as p
import jax
import jax3dp3
import jax3dp3 as j
import trimesh
import jax.numpy as jnp
import jax3dp3.pybullet_utils
import jax3dp3.transforms_3d as t3d
import pybullet_data
import pybullet_planning
import time
import jax

jax3dp3.setup_visualizer()

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


# p.connect(p.GUI)
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF


name = "004_sugar_box"
meshes = [
    trimesh.load(f"/home/nishadgothoskar/models/{name}/textured_simple.obj")
]
meshes = [
    jax3dp3.mesh.center_mesh(mesh) for mesh in meshes
]
for mesh in meshes:
    mesh.vertices *= 100.0

idx = 0
mesh = meshes[idx]
mesh_path = f"/tmp/{name}/{name}.obj"
os.makedirs(os.path.dirname(mesh_path),exist_ok=True)
mesh.export(mesh_path)

p.resetSimulation()
object, obj_dims = jax3dp3.pybullet_utils.add_mesh(mesh_path)
planeId = p.loadURDF("plane.urdf")


table_pose = jnp.eye(4)
table_dims = jnp.array([10.0, 10.0, 1e-10])


jax3dp3.clear()
jax3dp3.show_trimesh("1", meshes[idx])

iterations = 40
keys = jax.random.split(jax.random.PRNGKey(3), iterations)
object_poses = []

rots = j.make_rotation_grid_enumeration(20,10)

for i in range(len(rots)):
    # contact_params = jnp.array([0.0, 0.0, 0.0])
    # obj_pose = jax3dp3.scene_graph.pose_from_contact(
    #     contact_params,
    #     2, i,
    #     table_dims, obj_dims,
    #     table_pose
    # )
    # obj_pose = t3d.transform_from_pos(jnp.array([0.0, 0.0, 20.0])) @ j.distributions.vmf(keys[i], 0.1)
    obj_pose = t3d.transform_from_pos(jnp.array([0.0, 0.0, 20.0])) @ rots[i]

    pose = obj_pose
    jax3dp3.pybullet_utils.set_pose_wrapped(object, pose)
    jax3dp3.show_trimesh("1", meshes[idx])


    gravZ=-10
    p.setGravity(0, 0, gravZ)
    p.setTimeStep(0.05)

    for _ in range(300):
        p.stepSimulation()
        pose = jax3dp3.pybullet_utils.get_pose_wrapped(object)
        jax3dp3.set_pose("1", pose)


    object_poses.append(
        jax3dp3.pybullet_utils.get_pose_wrapped(object)
    )



contact_planes = []
for i in range(len(object_poses)):
    pose = object_poses[i]
    contact_point = jnp.eye(4)
    contact_point = contact_point.at[:2,3].set(pose[:2,3])
    contact_plane = (t3d.inverse_pose(pose) @ contact_point) @ t3d.transform_from_axis_angle(jnp.array([1.0, 1.0, 0.0]), jnp.pi)
    contact_planes.append(contact_plane)


jax3dp3.setup_visualizer()
jax3dp3.clear()
jax3dp3.show_trimesh("1", meshes[idx])
jax3dp3.set_pose("1", t3d.identity_pose())
for i in range(len(contact_planes)):
    jax3dp3.show_pose(f"contact_pose_{i}", 
        contact_planes[i],
        size=1.0
    )






from IPython import embed; embed()

rgb, depth, segmentation = jax3dp3.pybullet_utils.capture_image(cam_pose, h,w, fx,fy, cx,cy, near,far)
jax3dp3.viz.get_rgb_image(rgb).save("rgb.png")



p.resetDebugVisualizerCamera( cameraDistance=1, cameraYaw=30, cameraPitch=-52, cameraTargetPosition=[0,0,0])




rgbs = []
for _ in range(100):
    p.stepSimulation()
    rgb, depth, segmentation = jax3dp3.pybullet_utils.capture_image(cam_pose, h,w, fx,fy, cx,cy, near,far)
    rgbs.append(rgb)

from IPython import embed; embed()