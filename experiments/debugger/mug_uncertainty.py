
import jax3dp3
import jax3dp3 as j
import trimesh
import jax3dp3.transforms_3d as t3d
import jax.numpy as jnp
import os

jax3dp3.setup_visualizer()

state = jax3dp3.OnlineJax3DP3()
h, w, fx,fy, cx,cy = (
    300,
    300,
    200.0,200.0,
    150.0,150.0
)
near,far = 0.001, 5.0
camera_params = (h,w,fx,fy,cx,cy,near,far)

state.start_renderer(camera_params)

model_dir = os.path.join(jax3dp3.utils.get_assets_dir(), "bop/ycbv/models")
model_names = jax3dp3.ycb_loader.MODEL_NAMES
meshes = [trimesh.load(os.path.join(model_dir,"obj_" + f"{str(idx+1).rjust(6, '0')}.ply")) for idx in range(21)]
for (mesh, mesh_name) in zip(meshes, model_names):
    state.add_trimesh(mesh, mesh_name, mesh_scaling_factor=1.0/1000.0)

obj_pose = jnp.eye(4)
camera_pose = t3d.transform_from_rot_and_pos(
    t3d.rotation_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), -jnp.pi/10 - jnp.pi/2), 
    jnp.array([0.0, -0.3, 0.1])
)

obj_pose = obj_pose @ t3d.transform_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), jnp.pi/2)

obj_idx = 13

obj_pose_in_cam_frame = t3d.inverse_pose(camera_pose) @ obj_pose

img = j.render_single_object(obj_pose_in_cam_frame, 13)
j.viz.get_depth_image(img[:,:,2],max=far).save("depth.png")

j.show_cloud("1", t3d.point_cloud_image_to_points(img))

obs_point_cloud_image = state.process_depth_to_point_cloud_image(img[:,:,2])



segmentation_image = 1.0 * (img[:,:,2] > 0.0)
segmentation_id = 1.0
obs_image_masked, obs_image_complement = jax3dp3.get_image_masked_and_complement(
    obs_point_cloud_image, segmentation_image, segmentation_id, far
)

angles = jnp.linspace(0.0, jnp.pi*2, 100)
import jax
rotations = jax.vmap(t3d.transform_from_axis_angle,in_axes=(None, 0))(jnp.array([0.0,0.0, 1.0]), angles)
translations = j.enumerations.make_translation_grid_enumeration(
    -0.02,-0.02,0.0,
    0.02,0.02,0.0,
    5,5,1
)
pose_proposals = jnp.einsum("aij,bjk->abik", rotations, translations).reshape(-1, 4, 4)

# get best pose proposal
rendered_object_images = jax3dp3.render_parallel(t3d.inverse_pose(camera_pose) @ pose_proposals, obj_idx)[...,:3]
rendered_images = jax3dp3.splice_in_object_parallel(rendered_object_images, obs_image_complement)

r_sweep = jnp.array([0.01])
outlier_prob=0.1
outlier_volume=1.0
weights = jax3dp3.threedp3_likelihood_with_r_parallel_jit(
    obs_point_cloud_image, rendered_images, r_sweep, outlier_prob, outlier_volume
)[0,:]
probabilities = jax3dp3.utils.normalize_log_scores(weights)
print(probabilities.sort())


order = jnp.argsort(-probabilities)
images = []
NUM = 20
for i in order[:NUM]:
    img = j.viz.get_depth_image(rendered_object_images[i,:,:,2],max=far)
    images.append(img)
j.viz.multi_panel(images, labels=["{:0.2f}".format(p) for p in probabilities[order[:NUM]]]).save("depth.png")



j.clear()
for i in order[:NUM]:
    j.show_pose(f"p_{i}", pose_proposals[i])

j.clear()
for i in order[:NUM]:
    j.show_trimesh(f"p_{i}", meshes[obj_idx], opacity=0.3)
    j.set_pose(f"p_{i}", pose_proposals[i])
    


j.clear()
j.show_cloud("1", t3d.point_cloud_image_to_points(rendered_images[order[10]]))
j.show_cloud("2", t3d.point_cloud_image_to_points(obs_point_cloud_image),color=j.RED)







from IPython import embed; embed()

