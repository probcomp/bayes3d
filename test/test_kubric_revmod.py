import jax.numpy as jnp
import bayes3d as b
import trimesh
import os
import numpy as np
import trimesh
from tqdm import tqdm
from bayes3d._rendering.photorealistic_renderers.kubric_interface import render_many
import png2avi as p2a

# --- creating the ycb dir from the working directory
bop_ycb_dir = os.path.join(b.utils.get_assets_dir(), "bop/ycbv")
rgbd, gt_ids, gt_poses, masks = b.utils.ycb_loader.get_test_img('54', '1', bop_ycb_dir)



mesh_paths = []
offset_poses = []
heights = []
names = []
model_dir = os.path.join(b.utils.get_assets_dir(), "ycb_video_models/models")
for i in tqdm(gt_ids):
    mesh_path = os.path.join(model_dir, b.utils.ycb_loader.MODEL_NAMES[i],"textured.obj")
    m, pose = b.utils.mesh.center_mesh(trimesh.load(mesh_path), return_pose=True)
    bbox, _ = b.utils.aabb(m.vertices)
    heights.append(bbox[2]) # get z axis
    names.append(b.utils.ycb_loader.MODEL_NAMES[i])
    offset_poses.append(pose)
    mesh_paths.append(
        mesh_path
    )

# print heights
print('Object Heights:')
for i, name in enumerate(names):
    print(name + ': ' + str(heights[i]))


intrinsics = b.Intrinsics(
    rgbd.intrinsics.height, rgbd.intrinsics.width,
    rgbd.intrinsics.fx, rgbd.intrinsics.fx,
    rgbd.intrinsics.width/2, rgbd.intrinsics.height/2,
    rgbd.intrinsics.near, rgbd.intrinsics.far
)

poses = []
for i in range(len(gt_ids)):
    poses.append(
        gt_poses[i] @ b.t3d.inverse_pose(offset_poses[i])
    )
poses = jnp.array(poses)


# Note: the hardcoded object has to be upright - how to detect this automatically?
centered_item = 0 # hardcoded to be object 0

center_obj_basis = poses[centered_item]

obj_poses = jnp.einsum('jk,ikl->ijl',b.t3d.inverse_pose(center_obj_basis),poses)

frames = 20

#translate dome to put objects flush with ground
#translation = b.t3d.transform_from_pos(jnp.array([0,0,-1.5*heights[centered_item]/2.0])) #1.5 is a hack factor

#dome_pose = translation@obj_poses[centered_item,:,:] 

dome_pose = jnp.eye(4)

vids = 1 #10

for i in range(vids):
    # turn interpolation back to linear for debugging
    # spherical interpolation
    rgbds, cp, co = render_many(mesh_paths, obj_poses, dome_pose, cam_poses, intrinsics, frames,seed=i, scaling_factor=1.0, lighting=5.0, interpolation='spherical') # turn off the camera-pose warps for video

    im_dir = "ku_scene_vids_linear/frames"+str(i)
    if not os.path.exists(im_dir):
        os.makedirs(im_dir)

    for frame in range(frames):
        b.get_rgb_image(rgbds[frame].rgb).save(im_dir+"/image{:03d}.png".format(frame))


    p2a.save(image_folder = im_dir, video_name = 'ku_scene_vids_linear/linear'+str(i)+'.avi', fps=5)
    gt_poses = np.concatenate((cp, co),axis=1)
    np.savetxt('ku_scene_vids_linear/cam_pos_ori'+str(i)+'.txt', gt_poses)


#from IPython import embed; embed() 