

import jax3dp3
import jax3dp3 as j
import trimesh
import jax3dp3.transforms_3d as t3d
import jax3dp3.pybullet
import jax.numpy as jnp
import os
import pybullet as p
import trimesh
import numpy as np

jax3dp3.setup_visualizer()

h, w, fx,fy, cx,cy = (
    300,
    300,
    400.0,400.0,
    150.0,150.0
)
near,far = 0.001, 5.0


model_path = os.path.join(j.utils.get_assets_dir() ,"bop/ycbv/models/obj_000013.ply")
mesh = trimesh.load(model_path)
mesh.vertices = mesh.vertices / 1000.0
name= "mug"

camera_pose = t3d.transform_from_rot_and_pos(
    t3d.rotation_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), - jnp.pi/2 - jnp.pi/10), 
    jnp.array([0.0, -0.3, 0.1])
)
obj_pose = t3d.transform_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), -jnp.pi/4 - jnp.pi)


# p.connect(p.GUI)

# p.resetSimulation()


# os.makedirs(name,exist_ok=True)
# mesh.export(os.path.join(name, "textured.obj"))

# obj, dims = jax3dp3.pybullet.add_mesh(os.path.join(name, "textured.obj"))

# jax3dp3.pybullet.set_pose_wrapped(obj, jnp.eye(4))

# rgb, depth, segmentation = jax3dp3.pybullet.capture_image(
#     camera_pose,
#     h, w, fx,fy, cx,cy , near, far
# )
# j.viz.get_rgb_image(rgb).save("mug.png")

# jax3dp3.pybullet.set_pose_wrapped(obj, obj_pose)
# rgb, depth, segmentation = jax3dp3.pybullet.capture_image(
#     camera_pose,
#     h, w, fx,fy, cx,cy , near, far
# )
# j.viz.get_rgb_image(rgb).save("mug_hidden.png")

# np.savez("data.npz", depth=depth)


gt_depth =  np.load("data.npz")["depth"]
camera_params = (h,w,fx,fy,cx,cy,near,far)
state = jax3dp3.OnlineJax3DP3()
state.start_renderer(camera_params)
state.add_trimesh(mesh,"1")

# gt_depth = j.render_single_object(t3d.inverse_pose(camera_pose) @ obj_pose, 0)[:,:,2]

obs_point_cloud_image = state.process_depth_to_point_cloud_image(gt_depth)

segmentation_image = 1.0 * (gt_depth > 0.0)
segmentation_id = 1.0
obs_image_masked, obs_image_complement = jax3dp3.get_image_masked_and_complement(
    obs_point_cloud_image, segmentation_image, segmentation_id, far
)

angles = jnp.linspace(0.0, 2*jnp.pi, 100)
import jax
rotations = jax.vmap(t3d.transform_from_axis_angle,in_axes=(None, 0))(jnp.array([0.0,0.0, 1.0]), angles)
translations = j.enumerations.make_translation_grid_enumeration(
    -0.01,-0.01,0.0,
    0.01,0.01,0.0,
    5,5,1
)
pose_proposals = jnp.einsum("aij,bjk->abik", rotations, translations).reshape(-1, 4, 4)

# get best pose proposal
rendered_object_images = jax3dp3.render_parallel(t3d.inverse_pose(camera_pose) @ pose_proposals, 0)[...,:3]
rendered_images = jax3dp3.splice_in_object_parallel(rendered_object_images, obs_image_complement)

r_sweep = jnp.array([0.02])
outlier_prob=0.1
outlier_volume=1.0
weights = jax3dp3.threedp3_likelihood_with_r_parallel_jit(
    obs_point_cloud_image, rendered_images, r_sweep, outlier_prob, outlier_volume
)[0,:]
probabilities = jax3dp3.utils.normalize_log_scores(weights)
print(probabilities.sort())


order = jnp.argsort(-probabilities)

NUM = 30

alternate_camera_pose = t3d.transform_from_rot_and_pos(
    t3d.rotation_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), - jnp.pi/2 - jnp.pi/2), 
    jnp.array([0.0, 0.0, 0.2])
)
rendered_object_images = jax3dp3.render_parallel(t3d.inverse_pose(alternate_camera_pose) @ pose_proposals, 0)[order[:NUM],:,:,2]


images = []
for i in range(NUM):
    img = j.viz.get_depth_image(rendered_object_images[i],max=1.0)
    images.append(img)
j.viz.multi_panel(images, labels=["{:0.2f}".format(p) for p in probabilities[order[:NUM]]]).save("posterior.png")



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
ax.plot_wireframe(x, y, z, color=(0.0, 0.0, 0.0, 0.3), linewidths=0.3)


ax.axes.set_zticks([])
ax.axes.set_xticks([])
ax.axes.set_yticks([])

points = []
NUM = 1
for i in order[:NUM]:
    points.append(pose_proposals[i][:3,0])
points = np.array(points)

z = 0.1
for i in np.arange(.1,1.01,.1):
    ax.scatter(points[:,0], points[:,1], points[:,2], s=(50*i*(z*.9+.1))**2, color=(1,0,0,.5/i/10))

plt.tight_layout()
plt.savefig("sphere.png")



from IPython import embed; embed()