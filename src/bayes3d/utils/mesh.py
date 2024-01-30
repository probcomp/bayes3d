from itertools import product

import jax
import jax.numpy as jnp
import numpy as np
import trimesh

import bayes3d as j
import bayes3d.transforms_3d as t3d


def center_mesh(mesh, return_pose=False):
    _, pose = j.utils.aabb(mesh.vertices)
    shift = np.array(pose[:3, 3])
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
    with open(filename, "w") as f:
        f.write(
            trimesh.exchange.obj.export_obj(
                mesh, include_normals=True, include_texture=True
            )
        )


def make_cuboid_mesh(dimensions):
    mesh = trimesh.creation.box(dimensions, np.eye(4))
    return mesh


def make_voxel_mesh_from_point_cloud(point_cloud, resolution):
    poses = jax.vmap(j.t3d.transform_from_pos)(point_cloud)
    all_voxels = [
        trimesh.creation.box(np.array([resolution, resolution, resolution]), p)
        for p in poses
    ]
    final_mesh = trimesh.util.concatenate(all_voxels)
    return final_mesh


def make_marching_cubes_mesh_from_point_cloud(point_cloud, pitch):
    mesh = trimesh.voxel.ops.points_to_marching_cubes(point_cloud, pitch=pitch)
    return mesh


def open3d_mesh_to_trimesh(mesh):
    return trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles)
    )


def make_alpha_mesh_from_point_cloud(point_cloud, alpha):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.array(point_cloud))
    learned_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha
    )
    learned_mesh_trimesh = trimesh.Trimesh(
        vertices=np.asarray(learned_mesh.vertices),
        faces=np.asarray(learned_mesh.triangles),
    )
    return learned_mesh_trimesh


def make_table_mesh(
    table_width, table_length, table_height, table_thickness, table_leg_width
):
    table_face = trimesh.creation.box(
        np.array([table_width, table_length, table_thickness]),
        np.array(
            t3d.transform_from_pos(
                jnp.array([0.0, 0.0, table_height / 2.0 - table_thickness / 2.0])
            )
        ),
    )
    table_leg_height = table_height - table_thickness
    leg_dims = np.array([table_leg_width, table_leg_width, table_leg_height])
    leg_center = np.array(
        [table_width, table_length]
    ) / 2.0 - table_leg_width / 2.0 * np.ones(2)
    leg_xys = [
        np.multiply(leg_center, np.array(signs))
        for signs in product([-1, +1], repeat=len(leg_center))
    ]
    table_legs = [
        trimesh.creation.box(
            leg_dims,
            np.array(
                t3d.transform_from_pos(
                    np.array([x, y, table_leg_height / 2.0 - table_height / 2.0])
                )
            ),
        )
        for (x, y) in leg_xys
    ]
    table = trimesh.util.concatenate([table_face] + table_legs)
    return table


def point_cloud_image_to_trimesh(point_cloud_image):
    height, width, _ = point_cloud_image.shape

    def ij_to_index(i, j):
        return i * width + j

    def ij_to_faces(ij):
        return jnp.array(
            [
                [
                    ij_to_index(ij[0], ij[1]),
                    ij_to_index(ij[0] + 1, ij[1]),
                    ij_to_index(ij[0], ij[1] + 1),
                ],
                [
                    ij_to_index(ij[0] + 1, ij[1]),
                    ij_to_index(ij[0] + 1, ij[1] + 1),
                    ij_to_index(ij[0], ij[1] + 1),
                ],
            ]
        )

    jj, ii = jnp.meshgrid(jnp.arange(width - 1), jnp.arange(height - 1))
    indices = jnp.stack([ii, jj], axis=-1)
    faces = jax.vmap(ij_to_faces)(indices.reshape(-1, 2)).reshape(-1, 3)
    print(faces.shape)
    # vertices = point_cloud_first.reshape(-1,3)
    # mesh = trimesh.Trimesh(vertices, faces)
