import numpy as np
import jax3dp3 as j
import jax.numpy as jnp
import os
import trimesh
import jax3dp3.transforms_3d as t3d

results  = np.load("tracking_results.npz",allow_pickle=True)
data_stannis  = np.load("data_for_tracking_experiment.npz",allow_pickle=True)


data = np.load("data.npz")
rgb_images = data["rgb_images"]
depth_images = data["depth_images"]
poses = data["poses"]
poses = jnp.array(poses)

cam_pose = jnp.array(data["camera_pose"])
poses = t3d.inverse_pose(cam_pose) @ poses

h,w,fx,fy,cx,cy,near,far = data["camera_params"]
intrinsics = j.Intrinsics(int(h),int(w),fx,fy,cx,cy,near,far)

renderer = j.Renderer(intrinsics)

model_dir = "/home/nishadgothoskar/data/bop/ycbv/models"
for idx in range(1,22):
    renderer.add_mesh_from_file(os.path.join(model_dir,"obj_" + "{}".format(idx).rjust(6, '0') + ".ply"))

viz_images = []
all_particles = jnp.array(results["particles_list"])
for t in range(len(rgb_images)):
    print(t)
    obj_idx = 1
    particle_image = j.render_point_cloud(all_particles[t, 1,:, :3, -1], intrinsics, 2)
    particle_image_viz = j.viz.resize_image(j.viz.get_depth_image(particle_image[:,:,2] > 0.0,max=2.0), rgb_images[t].shape[0], rgb_images[t].shape[1])
    particle_image_rgba = np.array(particle_image_viz)
    
    rgba = np.array(rgb_images[t])
    mask = particle_image[:,:,-1]>0
    rgba[mask] =  particle_image_rgba[mask]

    particle_overlay_image_viz =  j.viz.get_rgb_image(rgba)
    particle_overlay_image_viz.save(f"imgs/{t}.png")
    viz_images.append(
        particle_overlay_image_viz
    )

j.viz.make_gif(viz_images, "out.gif")
