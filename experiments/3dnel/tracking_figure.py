import numpy as np
import jax3dp3 as j
import jax.numpy as jnp
import os
import trimesh
import jax3dp3.transforms_3d as t3d


image, gt_ids, gt_poses, masks = j.ycb_loader.get_test_img('50', '620', "/home/nishadgothoskar/data/bop/ycbv")

intrinsics = image.intrinsics

intrinsics = j.Intrinsics(
    intrinsics.height,
    intrinsics.width,
    intrinsics.fx,
    intrinsics.fy,
    intrinsics.cx,
    intrinsics.cy,
    1.0,
    10000.0
)

results  = np.load("tracking_bop.npz",allow_pickle=True)
renderer = j.Renderer(intrinsics)


model_dir = "/home/nishadgothoskar/data/bop/ycbv/models"
for idx in range(1,22):
    renderer.add_mesh_from_file(os.path.join(model_dir,"obj_" + "{}".format(idx).rjust(6, '0') + ".ply"))

rgb_images = results["rgb"]
all_particles = jnp.array(results["particles_list"])

j.viz.get_depth_image(renderer.render_multiobject(all_particles[0, :, 0, :, :], gt_ids)[:,:,2], max=intrinsics.far).save("img.png")

orig_h = intrinsics.height
orig_w = intrinsics.width
viz_images = []
for t in range(len(rgb_images)):
    print(t)
    obj_idx = 1
    particles_rendered = j.render_point_cloud(all_particles[t, obj_idx,:, :3, -1], intrinsics, 2)
    
    particle_image = jnp.zeros((*particles_rendered.shape[:2],4))
    
    reconstruction = j.viz.resize_image(j.viz.get_depth_image(renderer.render_multiobject(all_particles[t,:,0], gt_ids)[:,:,2], max=intrinsics.far),orig_h,orig_w)
    
    ids = []
    for _ in range(all_particles.shape[2]):
        ids.append(gt_ids[obj_idx])

    reconstruction_single_object = j.viz.resize_image(j.viz.get_depth_image(renderer.render_multiobject(all_particles[t, obj_idx, :], ids)[:,:,2], max=intrinsics.far),orig_h,orig_w)
    
    reconstruction_single_object

    d = j.overlay_image(j.get_rgb_image(rgb_images[t]), reconstruction_single_object, alpha=0.7)
    d = jnp.array(d)
    d = d.at[particles_rendered[:,:,2] > 0.0, :].set(jnp.array([0.0, 255.0, 0.0, 255.0]))
    d = j.get_rgb_image(d)

    images = [j.get_rgb_image(rgb_images[t]), d]

    # rgb.save(f"imgs/{t}_rgb.png")
    # # depth_viz.save(f"imgs/{t}_depth.png")
    # # depth_viz.save(f"imgs/{t}_depth.png")
    # particle_overlay_image_viz.save(f"imgs/{t}_particles.png")

    viz_image = j.viz.vstack_images(images)
    
    viz_image.save(f"imgs/{t}.png")
    viz_images.append(
        viz_image
    )

j.viz.make_gif(viz_images, "out.gif")
from IPython import embed; embed()

from IPython import embed; embed()