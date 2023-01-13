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


prism_1 = trimesh.creation.box(np.array([4.0, 1.5, 1.5]))
prism_1_path = os.path.join(jax3dp3.utils.get_assets_dir(), "rectangular_prism_1.obj")
jax3dp3.mesh.export_mesh(prism_1, prism_1_path)

prism_2 = trimesh.creation.box(np.array([3.0, 1.5, 1.5]))
prism_2_path = os.path.join(jax3dp3.utils.get_assets_dir(), "rectangular_prism_2.obj")
jax3dp3.mesh.export_mesh(prism_2, prism_2_path)

occluder = trimesh.creation.box(np.array([5.0, 0.1, 3.0]))
occluder_path = os.path.join(jax3dp3.utils.get_assets_dir(), "occulder.obj")
occluder.export(occluder_path,include_normals=True)
jax3dp3.mesh.export_mesh(occluder, occluder_path)

table_mesh = jax3dp3.mesh.center_mesh(jax3dp3.mesh.make_table_mesh(
    15.0,
    11.0,
    5.0,
    0.3,
    0.1
))
table_mesh_path  = "/tmp/table/table.obj"
os.makedirs(os.path.dirname(table_mesh_path),exist_ok=True)
jax3dp3.mesh.export_mesh(table_mesh, table_mesh_path)

all_pybullet_objects = []
box_dims = []

colors = [
    [214/255.0, 209/255.0, 197/255.0, 1.0],
    [224/255.0, 158/255.0, 72/255.0, 1.0],
    [113/255.0, 133/255.0, 189/255.0, 1.0],
    [113/255.0, 133/255.0, 189/255.0, 1.0],
]
paths = [table_mesh_path, occluder_path, prism_1_path, prism_2_path]
for(c,path) in zip(colors, paths):
    obj, obj_dims = jax3dp3.pybullet.add_mesh(path,color=c)
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
        [0.0, -3.0, -jnp.pi/2],
        [2.0, 0.0,  -jnp.pi/2],
        [0.0, 0.0,  -jnp.pi/2],
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
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ]),
    jnp.array([0.0, -15.0, 10.0])
).dot(t3d.transform_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), -jnp.pi/6 ))

rgb,depth,seg = jax3dp3.pybullet.capture_image(cam_pose, h,w, fx,fy, cx,cy, near,far)
rgb = np.array(rgb).reshape((h,w,4))
jax3dp3.viz.save_rgba_image(rgb,255.0,"rgb.png")

np.savez("data.npz", rgb=[rgb], depth=[depth], segmentation=[seg],
         params=(h,w,fx,fy,cx,cy,near,far),
)





from IPython import embed; embed()
