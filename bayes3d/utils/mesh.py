import trimesh
# weird new issue with trimesh in the latest versions
from trimesh import voxel as trimesh_voxel
import numpy as np
import bayes3d.transforms_3d as t3d
import bayes3d as j
import jax
import jax.numpy as jnp
from itertools import product

def center_mesh(mesh, return_pose=False):
    _, pose = j.utils.aabb(mesh.vertices)
    shift = np.array(pose[:3,3])
    mesh.vertices = mesh.vertices - shift
    if return_pose:
        return mesh, pose
    return mesh

def scale_mesh(mesh, scaling=1.0):
    mesh.vertices = mesh.vertices * scaling
    return mesh

def load_mesh(mesh_filename, scaling=1.0):
    mesh = trimesh.load(mesh_filename)
    mesh.vertices = mesh.vertices * scaling
    return mesh

def export_mesh(mesh, filename):
    normals = mesh.face_normals
    normals = mesh.vertex_normals
    with open(filename,"w") as f:
        f.write(trimesh.exchange.obj.export_obj(mesh, include_normals=True, include_texture=True))

def make_cuboid_mesh(dimensions):
    mesh = trimesh.creation.box(
        dimensions,
        np.eye(4)
    )
    return mesh

def make_voxel_mesh_from_point_cloud(point_cloud, resolution):
    poses = jax.vmap(j.t3d.transform_from_pos)(point_cloud)
    all_voxels = [
        trimesh.creation.box(np.array([resolution,resolution,resolution]), p) for p in poses
    ]
    final_mesh = trimesh.util.concatenate(all_voxels)
    return final_mesh

def make_marching_cubes_mesh_from_point_cloud(
    point_cloud,
    pitch
):
    mesh = trimesh_voxel.ops.points_to_marching_cubes(point_cloud, pitch=pitch)
    return mesh

def make_alpha_mesh_from_point_cloud(
    point_cloud,
    alpha
):
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(point_cloud))
    learned_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    learned_mesh_trimesh = trimesh.Trimesh(vertices=np.asarray(learned_mesh.vertices), faces=np.asarray(learned_mesh.triangles))
    return learned_mesh_trimesh

def make_table_mesh(
    table_width,
    table_length,
    table_height,
    table_thickness,
    table_leg_width
):

    table_face = trimesh.creation.box(
        np.array([table_width, table_length, table_thickness]),
        np.array(t3d.transform_from_pos(jnp.array([0.0, 0.0, table_height/2.0 - table_thickness/2.])))
    )
    table_leg_height = table_height-table_thickness
    leg_dims =  np.array([table_leg_width, table_leg_width, table_leg_height])
    leg_center = np.array([table_width, table_length])/2. - table_leg_width/2.0*np.ones(2)
    leg_xys = [np.multiply(leg_center, np.array(signs))
                for signs in product([-1, +1], repeat=len(leg_center))]
    table_legs = [
        trimesh.creation.box(
            leg_dims,
            np.array(t3d.transform_from_pos(np.array([x, y, table_leg_height/2. - table_height/2.0])))
        )
        for (x,y) in leg_xys
    ]
    table = trimesh.util.concatenate([table_face] + table_legs)
    return table