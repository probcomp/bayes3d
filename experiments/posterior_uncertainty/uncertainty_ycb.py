
import jax3dp3 as j 
import trimesh
import jax3dp3.transforms_3d as t3d
import jax.numpy as jnp
import os
import numpy as np


# image, gt_ids, gt_poses, masks = j.ycb_loader.get_test_img(
#     '48', '1', "/home/nishadgothoskar/data/bop/ycbv"
# )
# i = 2


image, gt_ids, gt_poses, masks = j.ycb_loader.get_test_img(
    '53', '1', "/home/nishadgothoskar/jax3dp3/assets/bop/ycbv"
)
i = 2

# data = np.load(os.path.join(j.utils.get_assets_dir(), "3dnel.npz"), allow_pickle=True)
# image = data["rgbd"].item()
# gt_ids = data["gt_ids"]
# gt_poses = data["gt_poses"]
# seg_ids = jnp.unique(image.segmentation)
# seg_ids = seg_ids[seg_ids != 0]
# masks = [image.segmentation == i for i in seg_ids]
# i = 4

rgb_viz = j.viz.get_rgb_image(image.rgb, 255.0)
rgb_viz.save("rgb.png")

intrinsics = j.camera.scale_camera_parameters(image.intrinsics, 0.3)
renderer = j.Renderer(intrinsics)

model_dir = "/home/nishadgothoskar/jax3dp3/assets/bop/ycbv/models"
mesh_paths = []
for idx in range(1,22):
    mesh_path = os.path.join(model_dir,"obj_" + "{}".format(idx).rjust(6, '0') + ".ply") 
    mesh_paths.append(mesh_path)
    renderer.add_mesh_from_file(
        mesh_path,
        scaling_factor=1.0/1000.0
    )
model_names = j.ycb_loader.MODEL_NAMES

obj_id = gt_ids[i]
segmentation_image = j.utils.resize(np.array(masks[i])*1.0, intrinsics.height, intrinsics.width)
segmentation_id = 1.0

mask = j.utils.resize((segmentation_image == segmentation_id)* 1.0, image.intrinsics.height, image.intrinsics.width)[...,None]

rgba = np.array(j.add_rgba_dimension(image.rgb))
rgba[mask[...,0] == 0,:] = 0.0
rgb_masked_viz = j.viz.get_rgb_image(rgba)
rgb_masked_viz.save("isolated.png")

rgb_masked_viz = np.array(rgb_masked_viz)

alpha=0.4
overlay_image = np.array(j.add_rgba_dimension(image.rgb) * alpha + (1-alpha)* rgb_masked_viz)
overlay_image[:,:,-1] = 255.0
overlay_image[overlay_image[:,:,0] <1.0, :] = 0.0
overlay = j.get_rgb_image(overlay_image)
overlay.save("masked_rgb.png")


obj_pose = image.camera_pose @ gt_poses[i]

reconstruction = j.resize_image(j.get_depth_image(renderer.render_single_object(t3d.inverse_pose(image.camera_pose) @ obj_pose, obj_id)[:,:,2], max=intrinsics.far), 
                                image.intrinsics.height,
                                image.intrinsics.width)
j.overlay_image(reconstruction, rgb_viz).save("reconstruction.png")

depth_scaled =  j.utils.resize(image.depth, intrinsics.height, intrinsics.width)
obs_point_cloud_image = t3d.unproject_depth(depth_scaled, intrinsics)
depth_masked, depth_complement = j.get_masked_and_complement_image(depth_scaled, segmentation_image, segmentation_id, intrinsics)
obs_point_cloud_image_masked = t3d.unproject_depth(depth_masked, intrinsics)
obs_point_cloud_image_complement = t3d.unproject_depth(depth_complement, intrinsics)


angles = jnp.linspace(0.0, 2*jnp.pi, 100)
import jax
rotations = jax.vmap(t3d.transform_from_axis_angle,in_axes=(None, 0))(jnp.array([0.0,0.0, 1.0]), angles)
translations = j.enumerations.make_translation_grid_enumeration(
    -0.01,-0.01,0.0,
    0.01,0.01,0.0,
    10,10,1
)
deltas = jnp.einsum("aij,bjk->abik", rotations, translations).reshape(-1, 4, 4)
pose_proposals = obj_pose @ deltas

r_sweep = jnp.array([0.01])
outlier_prob=0.2
outlier_volume=1.0

weights, fully_occluded_weight = j.c2f.score_poses(
    renderer,
    obj_id,
    obs_point_cloud_image,
    obs_point_cloud_image_complement,
    t3d.inverse_pose(image.camera_pose) @ pose_proposals,
    r_sweep,
    outlier_prob,
    outlier_volume,
)

probabilities = j.utils.normalize_log_scores(weights)
probabilities = probabilities.ravel()
print(probabilities.sort())


order = jnp.argsort(-probabilities)
NUM = (probabilities > 0.00001).sum()

rendered_object_images = renderer.render_parallel(t3d.inverse_pose(image.camera_pose) @ pose_proposals, obj_id)[...,:3]

images = []
for i in order[:NUM]:
    img = j.viz.get_depth_image(rendered_object_images[i][:,:,2],max=1.0)
    images.append(img)
j.viz.multi_panel(images, labels=["{:0.2f}".format(p) for p in probabilities[order[:NUM]]]).save("posterior.png")

# for i in order[:NUM]:
#     j.show_pose(f"{i}", pose_proposals[i])

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from itertools import product, combinations

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
# make the grid lines transparent
ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
x = np.cos(u)*np.sin(v)
y = np.sin(u)*np.sin(v)
z = np.cos(v)
ax.axes.set_xlim3d(-1.1, 1.1) 
ax.axes.set_ylim3d(-1.1, 1.1) 
ax.axes.set_zlim3d(-1.1, 1.1) 
ax.set_aspect("equal")
ax.plot_wireframe(x, y, z, color=(0.0, 0.0, 0.0, 0.3), linewidths=0.7)

ax.axes.set_zticks([])
ax.axes.set_xticks([])
ax.axes.set_yticks([])
ax.axes.set_axis_off()

points = []
for i in order[:NUM]:
    points.append(pose_proposals[i][:3,0])
points = np.array(points)

z = 0.1
for i in np.arange(.1,1.01,.1):
    ax.scatter(points[:,0], points[:,1], points[:,2], s=(50*i*(z*.9+.1))**2, color=(1,0,0,.5/i/10))

plt.tight_layout()
plt.savefig("sphere.png",transparent=True)


obj_pose_modified = np.array(j.t3d.inverse_pose(image.camera_pose) @ obj_pose)
obj_pose_modified[:3,3] *= 1000.0

intrinsics_modified = j.Intrinsics(
    image.intrinsics.height, image.intrinsics.width,
    image.intrinsics.fx, image.intrinsics.fy,
    image.intrinsics.cx, image.intrinsics.cy,
    1.0, 10000.0
)
viz = j.o3d_viz.O3DVis(intrinsics_modified)

viz.render.scene.clear_geometry()
viz.make_mesh(
    mesh_paths[obj_id], obj_pose_modified
)
light_dir = np.array([0.0, 0.0, 1.0])
viz.render.scene.scene.remove_light("light")
viz.render.scene.scene.add_directional_light('light',[1,1,1],light_dir,500000.0,True)

viz.render.scene.set_background(np.array([0.0, 0.0, 0.0, 0.0]))
img = viz.capture_image(image.intrinsics, np.eye(4))
transparent = img.at[img[:,:,0]<2.0, -1].set(0.0)
j.get_rgb_image(transparent).save("transparent.png")


# model_dir = "/home/nishadgothoskar/models"
# mesh_paths = []
# model_names = j.ycb_loader.MODEL_NAMES
# offset_poses = []
# for name in model_names:
#     mesh_path = os.path.join(model_dir,name,"textured.obj")
#     _, pose = j.mesh.center_mesh(trimesh.load(mesh_path), return_pose=True)
#     offset_poses.append(pose)
#     mesh_paths.append(
#         mesh_path
#     )

# import jax3dp3.kubric_interface
# rgb, seg, depth = jax3dp3.kubric_interface.render_kubric([mesh_paths[obj_id]], 
#     [
#         jnp.array([obj_pose @offset_poses[obj_id]])
#     ], jnp.eye(4), intrinsics, scaling_factor=1.0, lighting=5.0)



from IPython import embed; embed()