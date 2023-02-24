import blenderproc as bproc
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('data_file')
args = parser.parse_args()

data = np.load(args.data_file)
mesh_paths = data["mesh_paths"]
poses = data["poses"]
camera_pose = data["camera_pose"]
K = data["K"]
height = data["height"]
width = data["width"]
scaling_factor = data["scaling_factor"]

bproc.init()

for i in range(len(mesh_paths)):
    obj = bproc.loader.load_obj(mesh_paths[i])[0]
    for mat in obj.get_materials():
        mat.map_vertex_color()

    obj.set_local2world_mat(
        poses[i]
    )
    obj.set_scale([scaling_factor, scaling_factor, scaling_factor])
    obj.set_cp("category_id", i)


light = bproc.types.Light()
light.set_type("POINT")
light.set_location([0, 0, 0])
light.set_energy(1000)

bproc.camera.set_intrinsics_from_K_matrix(
    K, width, height
)

# Change coordinate frame of transformation matrix from OpenCV to Blender coordinates
cam2world = bproc.math.change_source_coordinate_frame_of_transformation_matrix(camera_pose, ["X", "-Y", "-Z"])
bproc.camera.add_camera_pose(cam2world)

# activate depth rendering
bproc.renderer.enable_depth_output(activate_antialiasing=False)

# render the whole pipeline
data = bproc.renderer.render()

bproc.writer.write_hdf5("/tmp/", data)
