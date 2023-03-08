import jax3dp3 as j
import jax.numpy as jnp
import numpy as np
import os
import trimesh
import copy
import open3d as o3d



viz_intrinsics = j.Intrinsics(
    1000,1000,
    1000.0,
    1000.0,
    500.0,
    500.0,
    0.01,
    20.0
)

viz = j.o3d_viz.O3DVis(viz_intrinsics)

viz.clear()

trimesh_mesh = trimesh.voxel.ops.points_to_marching_cubes(np.random.rand(100,3), pitch=0.5)
pose1 = j.t3d.transform_from_pos(jnp.array([-1.0, 0.0, 5.0]))
pose2 = j.t3d.transform_from_pos(jnp.array([0.0, 0.0, 4.0]))


viz.make_trimesh(trimesh_mesh, pose1, [0.0, 1.0, 0.0, 0.8])
viz.make_trimesh(trimesh_mesh, pose2, [1.0, 0.0, 0.0, 0.8])



viz.render.scene.set_lighting(viz.render.scene.LightingProfile.NO_SHADOWS, (0, 0, 0))
viz.set_background(np.array([0.0, 0.0, 0.0, 0.0]))
rgb = viz.capture_image(viz_intrinsics, np.eye(4))
rgb = rgb.at[(rgb[:,:,:3].sum(-1)) < 4, -1].set(0.0)
j.get_rgb_image(rgb).save("test_open3d_viz.png")


rgbd, gt_ids, gt_poses, masks = j.ycb_loader.get_test_img('52', '1', "/home/nishadgothoskar/data/bop/ycbv")

viz_intrinsics = j.Intrinsics(
    1000,1000,
    1000.0,
    1000.0,
    500.0,
    500.0,
    0.01,
    20.0
)

viz = j.o3d_viz.O3DVis(viz_intrinsics)

full_mask = jnp.array(masks).sum(0) > 0

viz.render.scene.clear_geometry()
cloud = j.t3d.apply_transform(j.t3d.unproject_depth(rgbd.depth, rgbd.intrinsics), rgbd.camera_pose)[full_mask].reshape(-1,3)
rgb_cloud = rgbd.rgb[full_mask,:3].reshape(-1,3)
viz.make_cloud(cloud, color=rgb_cloud / 255.0)

viz.make_camera(rgbd.intrinsics, rgbd.camera_pose, 0.1)

pos, target, up =(
    jnp.array([4.0, 0.0, 2.0]),
    jnp.array([0.0, 0.0, 0.0]),
    jnp.array([0.0, 0.0, 1.0]),
)
bystander_pose = j.t3d.transform_from_pos_target_up(pos, target, up)


j.get_rgb_image(viz.capture_image(rgbd.intrinsics, bystander_pose)).save("test_open3d_viz.png")



j.get_rgb_image(rgbd.rgb).save("rgb.png")
j.get_depth_image(rgbd.depth).save("depth.png")

viz.render.scene.clear_geometry()


pose = j.t3d.transform_from_pos(jnp.array([0.0, 0.0, 0.2]))
box = jnp.array([0.05, 0.04, 0.03])
viz.make_bounding_box(box, pose)

pose = j.t3d.transform_from_pos(jnp.array([0.0, 0.0, 0.2]))
cloud = np.random.rand(10000,3) / 20.0
moved_cloud = j.t3d.apply_transform(cloud, pose)

color = j.BLUE
colors = np.tile(color, (cloud.shape[0],1))

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(moved_cloud)
pcd.colors = o3d.utility.Vector3dVector(colors)


material = o3d.visualization.rendering.MaterialRecord()
material.shader = "defaultLit"

viz.render.scene.add_geometry("pcd7", pcd, material)

j.get_rgb_image(viz.capture_image(rgbd.intrinsics, np.eye(4))).save("test_open3d_viz.png")


model_dir = "/home/nishadgothoskar/models"
mesh_paths = []
model_names = j.ycb_loader.MODEL_NAMES
offset_poses = []
for name in model_names:
    mesh_path = os.path.join(model_dir,name,"textured.obj")
    mesh_paths.append(
        mesh_path
    )


import trimesh

idx = 1
mesh = trimesh.load(mesh_paths[idx])
viz.render.scene.clear_geometry()
pose = j.t3d.transform_from_pos(jnp.array([0.0, 0.0, 2.0]))
viz.make_mesh(mesh_paths[idx], pose)
viz.render.scene.set_lighting(viz.render.scene.LightingProfile.NO_SHADOWS, (0, 0, 0))

j.get_rgb_image(viz.capture_image(rgbd.intrinsics, np.eye(4))).save("test_open3d_viz.png")


import trimesh

j.meshcat.setup_visualizer()

trimesh_mesh = trimesh.voxel.ops.points_to_marching_cubes(np.random.rand(100,3), pitch=0.5)
j.meshcat.show_trimesh("1", trimesh_mesh)

def trimesh_to_o3d_triangle_mesh(trimesh_mesh):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices =  o3d.utility.Vector3dVector(trimesh_mesh.vertices)
    mesh.triangles = o3d.utility.Vector3iVector(trimesh_mesh.faces)
    mesh.triangle_normals = o3d.utility.Vector3dVector(np.array(trimesh_mesh.face_normals))
    return mesh



mesh1 = trimesh_to_o3d_triangle_mesh(trimesh_mesh)
mesh1.transform(j.t3d.transform_from_pos(jnp.array([0.0, 0.0, 4.0])))
mesh2 = trimesh_to_o3d_triangle_mesh(trimesh_mesh)
mesh2.transform(j.t3d.transform_from_pos(jnp.array([-1.0, 0.0, 5.0])))




# mtl = o3d.visualization.rendering.MaterialRecord()  # or MaterialRecord(), for later versions of Open3D
# mtl.base_color = [1.0, 1.0, 1.0, 1.0]  # RGBA
# mtl.shader = "defaultUnlit"

# mesh1.paint_uniform_color([1.0, 1.0, 0.0])

mtl = o3d.visualization.rendering.MaterialRecord()
mtl.shader = 'defaultLitTransparency'
mtl.base_color = [0.0, 1.0, 0.0, 0.7]


mtl2 = o3d.visualization.rendering.MaterialRecord()
mtl2.shader = 'defaultLitTransparency'
mtl2.base_color = [1.0, 0.0, 0.0, 1.0]

viz.render.scene.remove_geometry(f"1")
viz.render.scene.remove_geometry(f"2")
viz.render.scene.add_geometry(f"1", mesh1, mtl)
viz.render.scene.add_geometry(f"2", mesh2, mtl2)

viz.render.scene.set_lighting(viz.render.scene.LightingProfile.NO_SHADOWS, (0, 0, 0))
light_dir = np.array([0.0, 0.0, 1.0])
viz.render.scene.scene.remove_light('light2')
# viz.render.scene.scene.add_directional_light('light2',[1,1,1],light_dir,100000000.0,True)

j.get_rgb_image(viz.capture_image(viz_intrinsics, np.eye(4))).save("test_open3d_viz.png")


from IPython import embed; embed()