import numpy as np
import jax.numpy as jnp
import jax
import jax3dp3
import trimesh
import os

h, w, fx,fy, cx,cy = (
    300,
    300,
    200.0,200.0,
    150.0,150.0
)
near,far = 0.001, 50.0
r = 0.1
outlier_prob = 0.01
jax3dp3.setup_renderer(h, w, fx, fy, cx, cy, near, far)
mesh = trimesh.load(os.path.join(jax3dp3.utils.get_assets_dir(),"cube.obj"))
jax3dp3.load_model(mesh)

gt_poses = jnp.tile(jnp.array([
    [1.0, 0.0, 0.0, 0.0],   
    [0.0, 1.0, 0.0, -1.0],   
    [0.0, 0.0, 1.0, 8.0],   
    [0.0, 0.0, 0.0, 1.0],   
    ]
)[None,...],(5,1,1))
gt_poses = gt_poses.at[:,0,3].set(jnp.linspace(-2.0, 2.0, gt_poses.shape[0]))
gt_poses = gt_poses.at[:,2,3].set(jnp.linspace(10.0, 5.0, gt_poses.shape[0]))

max_depth = 15.0
multiobject_scene_img = jax3dp3.render_multiobject(gt_poses, [0 for _ in range(gt_poses.shape[0])])
jax3dp3.viz.save_depth_image(multiobject_scene_img[:,:,2], "gt_image.png", max=max_depth)


parallel_single_object_img = jax3dp3.render_multiobject_parallel(gt_poses[:,None, :,:],  [0])
jax3dp3.viz.save_depth_image(parallel_single_object_img[0,:,:,2], "img_1.png", max=max_depth)
jax3dp3.viz.save_depth_image(parallel_single_object_img[-1,:,:,2], "img_2.png", max=max_depth)

from IPython import embed; embed()


### Test masking in a multiobject scene

# get segmentation map
occx, occy = jnp.nonzero(multiobject_scene_img[:,:,2])
parallel_single_object_depth = np.array(parallel_single_object_img[:, :, :, 2])
parallel_single_object_depth[parallel_single_object_depth == 0] = float('inf')
get_segmentation_from_img_v = jax.vmap(lambda r,c: jnp.argmin(jnp.asarray(parallel_single_object_depth)[:,r,c]), in_axes=(0,0))

segmentation = np.ones(multiobject_scene_img[:,:,2].shape) * -1.0
segmentation[occx,occy] = get_segmentation_from_img_v(occx, occy)

# get depth data
depth_data = multiobject_scene_img[:,:,2]

# get gt complement data
test_obj = 1  # select which object to isolate via masking
gt_img_complement = jax3dp3.renderer.get_gt_img_complement(depth_data, segmentation, test_obj, h, w, fx, fy, cx, cy)
jax3dp3.viz.save_depth_image(gt_img_complement[:, :, 2], "gt_img_complement.png", max=far)

# render unmasked images
images_unmasked = parallel_single_object_img
unmasked = jax3dp3.viz.get_depth_image(
    images_unmasked[test_obj,:,:,2], max=max_depth  
)
unmasked.save("best_render_unmasked_1.png")

## render multiple masked images, produced by the mask
images = jax3dp3.renderer.get_masked_images(images_unmasked, gt_img_complement)
pred = jax3dp3.viz.get_depth_image(
    images[test_obj,:,:,2], max=max_depth   # visible portion of the test object should be isolated by the mask
)
pred.save("best_render_masked_1.png")


# render single masked image from single unmasked image
image = jax3dp3.renderer.get_single_masked_image(images_unmasked[test_obj], gt_img_complement)
pred = jax3dp3.viz.get_depth_image(
    image[:,:,2], max=max_depth
)
pred.save("best_render_masked_2.png")  # should be identical to best_render_1


from IPython import embed; embed()