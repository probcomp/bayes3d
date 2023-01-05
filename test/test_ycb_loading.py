import jax.numpy as jnp
import jax3dp3
from jax3dp3.data import BOPTestImage, get_test_img
from jax3dp3.viz import save_depth_image, get_depth_image
import trimesh


test_img = get_test_img('54', '1568')
h, w = test_img.get_image_dims()

fx, fy, cx, cy = test_img.get_camera_intrinsics()
print("intrinsics:", h, w, fx, fy, cx, cy)
h, w, fx, fy, cx, cy  = jax3dp3.camera.scale_camera_parameters(h,w,fx,fy,cx,cy, 0.5)
print("intrinsics:", h, w, fx, fy, cx, cy)
near = 1.0; far = 50.0

jax3dp3.setup_renderer(h, w, fx, fy, cx, cy, near, far)


model_object_names = os.listdir(os.path.join(jax3dp3.utils.get_assets_dir(),"bop_models"))
model_dir = os.path.join(jax3dp3.utils.get_assets_dir(),"models")
model_names = os.listdir(model_dir)
for model_name in model_names:
    if model_name[-4:] != ".ply":
        continue
    mesh = trimesh.load(os.path.join(jax3dp3.utils.get_assets_dir(),f"{model_name}"))
    jax3dp3.load_model(mesh, h, w)


cam_pose = test_img.get_camera_pose()
object_poses = test_img.get_gt_poses()
object_ids = test_img.get_gt_indices()

for pose, id in zip(object_poses, object_ids):
    pose_in_cam_frame = jnp.linalg.inv(cam_pose).dot(pose)
    gt_image = jax3dp3.render(pose_in_cam_frame, h,w, id)

save_depth_image(gt_image[:,:,2], "render_test.png", min=near, max=far)
save_depth_image(test_img.get_depth_image(), "render_gt.png", min=near, max=far)

from IPython import embed; embed()
