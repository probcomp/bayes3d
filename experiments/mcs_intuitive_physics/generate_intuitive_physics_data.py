import machine_common_sense as mcs
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

controller = mcs.create_controller(config_file_or_dict='./config.ini')
scene_data = mcs.load_scene_json_file("scene.json")

data = [ controller.start_scene(scene_data) ]

get_available_actions  = lambda output: [i[0] for i in output.action_list]
while True:
    output = controller.step("Pass")
    data.append(output)
    if "Pass" not in get_available_actions(output):
        break


###### Camera intrinsics ######
step_metadata = data[0]
width, height = step_metadata.camera_aspect_ratio
aspect_ratio = width / height

# Camera principal point is the center of the image.
cx, cy = width / 2.0, height / 2.0
# Vertical field of view is given.
fov_y = np.deg2rad(step_metadata.camera_field_of_view)
# Convert field of view to distance to scale by aspect ratio and
# convert back to radians to recover the horizontal field of view.
fov_x = 2 * np.arctan(aspect_ratio * np.tan(fov_y / 2.0))
# Use the following relation to recover the focal length:
#   FOV = 2 * atan( (0.5 * IMAGE_PLANE_SIZE) / FOCAL_LENGTH )
fx = cx / np.tan(fov_x / 2.0)
fy = cy / np.tan(fov_y / 2.0)
clipping_near, clipping_far = step_metadata.camera_clipping_planes

K = np.array([
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0],
])


rgb_images = []
for o in data:
    rgb_images.append(np.array(o.image_list[0]))
rgb_images = np.array(rgb_images)

depth_images = []
for o in data:
    depth_images.append(np.array(o.depth_map_list[0]))
depth_images = np.array(depth_images)

np.savez("data.npz", rgb_images=rgb_images, depth_images=depth_images, fx=fx, fy=fy, cx=cx, cy=cy) 

max_depth = 30.0
middle_width = 20
cm = plt.get_cmap("turbo")
images = []
original_width = depth_images.shape[2]
original_height = depth_images.shape[1]
for i in range(depth_images.shape[0]):
    dst = Image.new(
        "RGBA", (2 * original_width + 1*middle_width, original_height)
    )

    rgb = rgb_images[i]
    rgb_img = Image.fromarray(
        rgb.astype(np.int8), mode="RGB"
    )
    dst.paste(
        rgb_img,
        (0,0)
    )

    dst.paste(
        Image.fromarray(
            np.rint(
                cm(np.array(depth_images[i, :, :]) / max_depth) * 255.0
            ).astype(np.int8),
            mode="RGBA",
        ).resize((original_width,original_height)),
        (original_width + middle_width, 0),
    )
    images.append(dst)


images[0].save(
    fp="out.gif",
    format="GIF",
    append_images=images,
    save_all=True,
    duration=100,
    loop=0,
)

from IPython import embed; embed()

