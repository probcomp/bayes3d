import jax3dp3 as j
import os
import torch
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, build_sam

bop_ycb_dir = os.path.join(j.utils.get_assets_dir(), "bop/ycbv")
rgbd, gt_ids, gt_poses, masks = j.ycb_loader.get_test_img('52', '1', bop_ycb_dir)


sam = build_sam(checkpoint="/home/nishadgothoskar/jax3dp3/assets/sam/sam_vit_h_4b8939.pth")
sam.to(device="cuda")

mask_generator = SamAutomaticMaskGenerator(sam)
boxes= mask_generator.generate(np.array(rgbd.rgb))
full_segmentation = jnp.zeros(rgbd.rgb.shape[:2])
for i in range(len(boxes)):
    seg_mask = jnp.array(boxes[i]["segmentation"])
    j.get_depth_image(seg_mask).save(f"{i}.png") 
    full_segmentation = full_segmentation.at[seg_mask].set(i+1)

segmentation_image = j.get_depth_image(full_segmentation,max=full_segmentation.max())
segmentation_image.save("seg.png")


# sam = build_sam()
# sam.to(device="cuda")
# mask_generator = SamAutomaticMaskGenerator(sam)

j.get_rgb_image(rgbd.rgb).save("rgb.png")
mask_generator.generate(np.array(rgbd.rgb))

mask_generator.generate(np.array(img))

sam = sam_model_registry["default"](checkpoint=args.checkpoint)
_ = sam.to(device=args.device)
output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
amg_kwargs = get_amg_kwargs(args)
generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)


mask_generator = SamAutomaticMaskGenerator["default"](build_sam())


