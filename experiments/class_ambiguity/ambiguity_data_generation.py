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

p.connect(p.GUI)
# p.connect(p.DIRECT)


prism_1 = trimesh.creation.box(np.array([2.0, 0.5, 0.5]))
prism_1_path = os.path.join(jax3dp3.utils.get_assets_dir(), "rectangular_prism_1.obj")
prism_1.export(prism_1_path)

prism_2 = trimesh.creation.box(np.array([3.0, 0.5, 0.5]))
prism_2_path = os.path.join(jax3dp3.utils.get_assets_dir(), "rectangular_prism_2.obj")
prism_2.export(prism_2_path)

occluder = trimesh.creation.box(np.array([3.0, 0.1, 3.0]))
occluder_path = os.path.join(jax3dp3.utils.get_assets_dir(), "occulder.obj")
occluder.export(occluder_path)

table_mesh = jax3dp3.mesh.center_mesh(jax3dp3.mesh.make_table_mesh(
    10.0,
    6.0,
    5.0,
    0.3,
    0.1
))
table_mesh_path  = "/tmp/table/table.obj"
os.makedirs(os.path.dirname(table_mesh_path),exist_ok=True)
table_mesh.export(table_mesh_path)

all_pybullet_objects = []
box_dims = []

for path in [table_mesh_path, occluder_path, prism_1_path, prism_2_path]:
    obj, obj_dims = jax3dp3.pybullet.add_mesh(path,color=[1.0, 0.0, 0.0, 1.0])
    all_pybullet_objects.append(obj)
    box_dims.append(obj_dims)

box_dims = jnp.array(box_dims)


absolute_poses = jnp.array([
    t3d.transform_from_pos(jnp.array([0.0, 0.0, box_dims[0,2]/2.0])),
    jnp.eye(4),
    jnp.eye(4),
    jnp.eye(4)
])

edges = jnp.array([
    [-1,0],
    [0,1],
    [0,2],
    [0,3],
])

contact_params = jnp.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, -jnp.pi/2],
        [0.2, 0.2,  0.0],
        [0.2, 0.2,  0.0],
    ]
)

parent_face_params = jnp.array([2,2,2,2])
child_face_params = jnp.array([3,3,3,3])

get_poses_jit = jax.jit(jax3dp3.scene_graph.absolute_poses_from_scene_graph)
poses = get_poses_jit(
    absolute_poses, box_dims, edges, contact_params, parent_face_params, child_face_params
)
for (obj, pose) in zip(all_pybullet_objects, poses):
    jax3dp3.pybullet.set_pose_wrapped(obj, pose)


cam_pose = t3d.transform_from_rot_and_pos(
    jnp.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, -1.0, 0.0],
    ]),
    jnp.array([0.0, 15.0, 10.0])
).dot(t3d.transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), -jnp.pi/6 ))

rgb, depth, segmentation = jax3dp3.pybullet.capture_image(
    cam_pose,
    h, w, fx,fy, cx,cy , near, far
)
jax3dp3.viz.save_rgba_image(rgb,255.0,"rgb.png")

from IPython import embed; embed()

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
