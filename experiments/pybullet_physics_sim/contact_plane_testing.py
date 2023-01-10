import os
import numpy as np
import pybullet as p
import jax
import jax3dp3
import trimesh
import jax.numpy as jnp
import jax3dp3.pybullet
import jax3dp3.transforms_3d as t3d
import pybullet_data
import time

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
p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF

model_dir = os.path.join(jax3dp3.utils.get_assets_dir(),"models")
model_names = os.listdir(model_dir)
print(model_names)
idx = 17
cracker_box_path = os.path.join(jax3dp3.utils.get_assets_dir(), "models/{}/textured_simple.obj".format(model_names[idx]))
cracker_box_mesh = trimesh.load(cracker_box_path)
cracker_box_mesh_centered = jax3dp3.mesh.center_mesh(cracker_box_mesh)
cracker_box_centered_path = "/tmp/obj1/obj.obj"
os.makedirs(os.path.dirname(cracker_box_centered_path),exist_ok=True)
cracker_box_mesh_centered.export(cracker_box_centered_path)



for i in range(6):
    print(i)
    p.resetSimulation()
    object, obj_dims = jax3dp3.pybullet.add_mesh(cracker_box_centered_path)

    table_pose = jnp.eye(4)
    table_dims = jnp.array([10.0, 10.0, 1e-6])

    contact_params = jnp.array([0.0, 0.0, 0.0])
    obj_pose = jax3dp3.scene_graph.pose_from_contact(
        contact_params,
        2, i,
        table_dims, obj_dims,
        table_pose
    )
    jax3dp3.pybullet.set_pose_wrapped(object, obj_pose)

    p.resetDebugVisualizerCamera( cameraDistance=1, cameraYaw=30, cameraPitch=-52, cameraTargetPosition=[0,0,0])
    planeId = p.loadURDF("plane.urdf")

    gravZ=-10
    p.setGravity(0, 0, gravZ)
    p.setTimeStep(0.01)


    for _ in range(100):
        p.stepSimulation()
        time.sleep(0.05)

from IPython import embed; embed()