import os
import numpy as np
import pybullet as p
import jax
import jax3dp3 as j
import jax3dp3.mesh
import jax.numpy as jnp
import jax3dp3.pybullet
import jax3dp3.transforms_3d as t3d
import trimesh
import pybullet as p
import pickle 


from collections import namedtuple

p.connect(p.DIRECT)
pybullet_objects = []

YCB_NAMES = jax3dp3.ycb_loader.MODEL_NAMES
model_dir = os.path.join(jax3dp3.utils.get_assets_dir(), "ycb_obj/models")

bop_ycb_dir = os.path.join(j.utils.get_assets_dir(), "bop/ycbv")
rgbd, gt_ids, gt_poses, masks = j.ycb_loader.get_test_img('52', '1', bop_ycb_dir)
intrinsics = rgbd.intrinsics


# setup renderer
model_dir = os.path.join(j.utils.get_assets_dir(), "ycb_video_models/models")
print(f"{model_dir} exists: {os.path.exists(model_dir)}")
model_names = j.ycb_loader.MODEL_NAMES
IDX = 13
print(model_names[IDX])
mesh_path = os.path.join(model_dir,YCB_NAMES[IDX],"textured.obj")
obj, obj_dims = jax3dp3.pybullet.add_mesh(mesh_path)
pybullet_objects.append(obj)

cam_pose = j.t3d.transform_from_pos_target_up(
    jnp.array([0.0, 1.0, 0.0]),
    jnp.array([0.0, 0.0, 0.0]),
    jnp.array([0.0, 0.0, 1.5]),
)
jax3dp3.pybullet.set_pose_wrapped(obj, jax3dp3.distributions.gaussian_vmf(jax.random.PRNGKey(1), 0.00001, 0.001))

rgb,depth,seg = jax3dp3.pybullet.capture_image(cam_pose, intrinsics.height, intrinsics.width, intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy, intrinsics.near, intrinsics.far)

jax3dp3.viz.get_rgb_image(rgb).save("mug_rgb.png")
jax3dp3.viz.get_depth_image(depth, min=np.min(depth), max=np.max(depth)).save(f"mug_depth.png")


jax3dp3.pybullet.remove_body(pybullet_objects[-1])


with open(f"mug1.pkl", 'wb') as file:
    pickle.dump({'rgb':rgb, 'depth':depth, 'intrinsics':intrinsics}, file)


from IPython import embed; embed()

