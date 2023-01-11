# TODO simplifiy this @karen

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