
struct Mesh
    pyobj::PyObject
end

struct VoxelGrid
    occupied::BitArray{3}
end

function load_ycb_mesh(idx::Int; scale=1000.0)
    mesh_path = j.utils.get_assets_dir() * "/bop/ycbv/models/obj_$(lpad(idx, 6, "0")).ply"
    pymesh = trimesh.load(mesh_path)
    pymesh.vertices *= 1.0/scale
    pymesh = j.mesh.center_mesh(pymesh)
    Mesh(pymesh)
end

function make_mesh(shape::VoxelGrid; scale=1000.0, voxel_size=1.0)
    occupied_centers = voxel_size * vcat(map(cidx->[cidx.I...]', findall(shape.occupied))...)
    pymesh = trimesh.voxel.ops.points_to_marching_cubes(occupied_centers, pitch=0.1)
    pymesh.vertices *= 1.0/scale
    pymesh = j.mesh.center_mesh(pymesh)
    Mesh(pymesh)
end