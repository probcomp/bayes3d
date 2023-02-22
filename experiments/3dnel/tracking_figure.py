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


state =j.OnlineJax3DP3()
state.start_renderer(data["camera_params"], scaling_factor=0.3)
(h,w,fx,fy,cx,cy,near,far) = state.camera_params
orig_h, orig_w = data["camera_params"][:2]
orig_h, orig_w = int(orig_h), int(orig_w)


h, w, fx,fy, cx,cy = (
    480,
    640,
    500.0,500.0,
    320.0,240.0
)
near,far = 10.0, 2000.0


state.start_renderer((h, w, fx,fy, cx,cy, near, far), scaling_factor=1.0)

top_level_dir = j.utils.get_assets_dir()
model_names = ["cracker_box", "sugar_box"]
model_paths = [
    os.path.join(top_level_dir,"003_cracker_box/textured.obj"),
    os.path.join(top_level_dir,"004_sugar_box/textured.obj"),
]
for (path, name) in zip(model_paths,model_names):
    mesh = trimesh.load(path)
    mesh = j.mesh.center_mesh(mesh)
    state.add_trimesh(
        mesh, mesh_name=name, mesh_scaling_factor=1000.0
    )

all_particles = jnp.array(results["particles_list"])
j.viz.get_depth_image(j.render_multiobject(all_particles[0, :, 0], [0,1])[:,:,2], max=10.0).save("img.png")


viz_images = []
for t in range(len(rgb_images)):
    obj_idx = 1
    particle_image = j.render_point_cloud(all_particles[t, 1,:, :3, -1], h, w, fx,fy, cx,cy, near, far, 1)
    particle_image_viz = j.viz.resize_image( j.viz.get_depth_image(particle_image[:,:,2] > 0.0,max=2.0),orig_h,orig_w)

    reconstruction = j.viz.resize_image(j.viz.get_depth_image(j.render_multiobject(all_particles[t,:,0], [0,1])[:,:,2], max=far),orig_h,orig_w)
    
    reconstruction_single_object = j.viz.resize_image(j.viz.get_depth_image(j.render_multiobject(all_particles[t, obj_idx, :], [obj_idx for _ in range(400)])[:,:,2], max=far),orig_h,orig_w)
    
    rgb =  j.viz.get_rgb_image(rgb_images[t],255.0)
    # depth_viz = j.viz.resize_image(j.viz.get_depth_image(depth_images[t,:,:],max=far),orig_h,orig_w)

    particle_overlay_image_viz = j.viz.overlay_image(rgb, particle_image_viz,alpha=0.8)

    # rgb.save(f"imgs/{t}_rgb.png")
    # # depth_viz.save(f"imgs/{t}_depth.png")
    # # depth_viz.save(f"imgs/{t}_depth.png")
    # particle_overlay_image_viz.save(f"imgs/{t}_particles.png")

    # j.viz.overlay_image(rgb, reconstruction_single_object).save(f"imgs/{t}_overlay.png")
    images = [rgb, particle_overlay_image_viz]
    viz_image = j.viz.vstack_images(images)
    viz_image.save(f"imgs/{t}.png")
    viz_images.append(
        viz_image
    )

j.viz.make_gif(viz_images, "out.gif")

from IPython import embed; embed()