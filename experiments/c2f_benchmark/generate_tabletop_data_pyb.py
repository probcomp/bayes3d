import os
import numpy as np
import pybullet as p
import jax
import jax3dp3
import jax3dp3.mesh
import jax.numpy as jnp
import jax3dp3.pybullet
import jax3dp3.transforms_3d as t3d
import pickle

h, w, fx,fy, cx,cy = (
    600,
    600,
    500.0,500.0,
    320.0,240.0
)
near,far = 0.001, 20.0
r = 0.1
outlier_prob = 0.01
K = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])

max_depth = 15.0

p.connect(p.DIRECT)

height = 0.0
thickness = 1e-10
table_mesh = jax3dp3.mesh.center_mesh(jax3dp3.mesh.make_table_mesh(
    5.0,
    5.0,
    height,
    thickness,
    0.0
))

YCB_NAMES = jax3dp3.ycb_loader.MODEL_NAMES

# table_mesh_path  = "/tmp/table/table.obj"
# table_color = [214/255.0, 209/255.0, 197/255.0, 1.0]
# os.makedirs(os.path.dirname(table_mesh_path),exist_ok=True)
# jax3dp3.mesh.export_mesh(table_mesh, table_mesh_path)

# paths = [table_mesh_path, occluder_path, prism_1_path, prism_2_path]
model_dir = os.path.join(jax3dp3.utils.get_assets_dir(), "ycb_obj/models")



def get_face_params(ycb_obj_name):  # TODO: get valid face params for all ycb objs
    all_faces = [0,1,2,3,4,5]
    fdict = {
        "003_cracker_box":all_faces,
        "004_sugar_box":all_faces,
        "005_tomato_soup_can":[2,3],
        "010_potted_meat_can":[0,1,2,3] ,
        "011_banana":[2,3],
        "019_pitcher_base":[0,1,3],
        "021_bleach_cleaner":[0,1,3],
        "035_power_drill":[1,2,3],
        "036_wood_block": all_faces,
        "040_large_marker":[1,2,3,4,5],
        "051_large_clamp":[2,3] 
        }
    
    if ycb_obj_name not in fdict:
        return all_faces
    return fdict[ycb_obj_name]
    


def get_tabletop_image(y,x,rot,face_param,box_dims, all_pybullet_objects, save_img_dir, save_data_dir, ycb_obj_name,cnt=0):
    contact_param = jnp.array([y, x, rot])
    contact_plane = table_pose
    box_pose = jax3dp3.scene_graph.pose_from_contact_and_face_params(contact_param, face_param, box_dims[1], contact_plane) 

    # world frame
    # poses = [table_pose, box_pose @ table_pose]
    poses = [box_pose @ table_pose]


    for (obj, pose) in zip(all_pybullet_objects, poses):
        jax3dp3.pybullet.set_pose_wrapped(obj, pose)

    rgb,depth,seg = jax3dp3.pybullet.capture_image(camera_pose, h,w, fx,fy, cx,cy, near,far)
    seg = np.array(seg).reshape((h,w))
    rgb = np.array(rgb).reshape((h,w,4))
    depth = np.array(depth).reshape((h,w))
    jax3dp3.viz.save_rgba_image(rgb,255.0,f"{save_img_dir}/rgb_{cnt}_{np.round(y,3)}_{np.round(x,3)}_{face_param}.png")
    # jax3dp3.viz.save_depth_image(depth,max=far,filename=f"{save_img_dir}/depth_{cnt}_{np.round(y,3)}_{np.round(x,3)}_{face_param}.png")


    with open(f"{save_data_dir}/data_{cnt}.pkl", 'wb') as file:
        pickle.dump(
            {'rgb': rgb,
            'depth': depth,
            'segmentation':seg,
            'factor_depth': 1,
            'intrinsics':K,
            'camera_pose': np.asarray(camera_pose),
            'object_pose': np.asarray(poses[-1]),
            'contact_param': np.asarray(contact_param),
            'contact_face': face_param,
            'contact_plane': np.asarray(table_pose),
            'object_name': ycb_obj_name
            }, file
        )

    print(f"{cnt}  {contact_param}")

    # from IPython import embed; embed()


def get_singleobj_test_data(ycb_obj_name,cnt=0):
    all_pybullet_objects = []
    box_dims = []

    obj_path = os.path.join(model_dir, os.path.join(ycb_obj_name, 'textured_simple.obj'))

    # paths = [table_mesh_path, obj_path]
    paths = [obj_path]

    for i in range(len(paths)):
        path = paths[i]
        # if i == 0:  # table
        #     obj, obj_dims = jax3dp3.pybullet.add_mesh(path, color=table_color)
        # else:
        obj, obj_dims = jax3dp3.pybullet.add_mesh(path, center=True)
        all_pybullet_objects.append(obj)
        box_dims.append(obj_dims)

    box_dims = jnp.array(box_dims)

    ## Generate same pose from contact params
    save_dir = "test_data"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    if not os.path.exists(f"{save_dir}/imgs"):
        os.mkdir(f"{save_dir}/imgs")
    if not os.path.exists(f"{save_dir}/data"):
        os.mkdir(f"{save_dir}/data")

    save_img_dir = f"{save_dir}/imgs/" + ycb_obj_name
    save_data_dir = f"{save_dir}/data/" + ycb_obj_name
    if not os.path.exists(save_img_dir):
        os.mkdir(save_img_dir)
        os.mkdir(save_data_dir)

    np.random.seed(1222)
    
    for i, y in enumerate(ys):
        for x in xs[i]:
            for face_param in get_face_params(ycb_obj_name):

                rot = np.random.uniform(-jnp.pi, jnp.pi)
                get_tabletop_image(y,x,rot,face_param,box_dims, all_pybullet_objects, save_img_dir, save_data_dir, ycb_obj_name,cnt)
                cnt += 1

    # remove current obj from pybullet simulation for next object processing
    jax3dp3.pybullet.remove_body(all_pybullet_objects[-1])

    return cnt



# YCB_NAMES = ["003_cracker_box"]
for obj_name in YCB_NAMES:
    print("\n=====================================")
    print("Generating data for ", obj_name)
    print("=====================================\n")

    cnt = 0
    small_item_names = ["005_tomato_soup_can", "040_large_marker"]

    if obj_name in small_item_names: 
        for setting in [1,2]:
            if setting == 1:
                camera_pose = t3d.transform_from_rot_and_pos(
                    t3d.rotation_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), -jnp.pi/2 - jnp.pi/8),
                    jnp.array([0.0, -1.0, 0.8])
                )
                turn = t3d.transform_from_rot(t3d.rotation_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), jnp.pi/2))
                camera_pose = turn @ camera_pose

            if setting == 2:
                camera_pose = t3d.transform_from_rot_and_pos(
                    t3d.rotation_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), -jnp.pi/2 - jnp.pi/4),
                    jnp.array([0.0, -0.8, 0.8])
                )
                turn = t3d.transform_from_rot(t3d.rotation_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), jnp.pi/2))
                camera_pose = turn @ camera_pose
            ys = np.linspace(0.35,0.45,3)
            far_x = np.linspace(-0.35,0.35,2)
            near_x = np.linspace(-0.1, 0.1,2)

            xs = [far_x,near_x,near_x]

            table_pose = jnp.eye(4)

            cnt = get_singleobj_test_data(obj_name,cnt)

    else:
        for setting in [1,2]:
            if setting == 1:
                camera_pose = t3d.transform_from_rot_and_pos(
                    t3d.rotation_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), -jnp.pi/2 - jnp.pi/8),
                    jnp.array([0.0, -1.0, 0.8])
                )
                turn = t3d.transform_from_rot(t3d.rotation_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), jnp.pi/2))
                camera_pose = turn @ camera_pose
                ys = np.linspace(0.15,0.45,3)
                far_x = np.linspace(-0.45,0.45,2)
                near_x = np.linspace(-0.15, 0.15,2)

            if setting == 2:
                camera_pose = t3d.transform_from_rot_and_pos(
                    t3d.rotation_from_axis_angle(jnp.array([1.0, 0.0, 0.0]), -jnp.pi/2 - jnp.pi/4),
                    jnp.array([0.0, -0.8, 0.8])
                )
                turn = t3d.transform_from_rot(t3d.rotation_from_axis_angle(jnp.array([0.0, 0.0, 1.0]), jnp.pi/2))
                camera_pose = turn @ camera_pose
                ys = np.linspace(0.05,0.45,3)
                far_x = np.linspace(-0.45,0.45,2)
                near_x = np.linspace(-0.15, 0.15,2)


            xs = [far_x,near_x,near_x]

            table_pose = jnp.eye(4)

            cnt = get_singleobj_test_data(obj_name,cnt)

