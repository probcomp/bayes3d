import os
import pickle
import sys
import warnings

import bayes3d as j
import jax.numpy as jnp
import numpy as np
from segment_anything import SamAutomaticMaskGenerator, build_sam

sys.path.extend(["/home/nishadgothoskar/ptamp/pybullet_planning"])
sys.path.extend(["/home/nishadgothoskar/ptamp"])
warnings.filterwarnings("ignore")


bop_ycb_dir = os.path.join(j.utils.get_assets_dir(), "bop/ycbv")
rgbd, gt_ids, gt_poses, masks = j.ycb_loader.get_test_img("52", "1", bop_ycb_dir)

test_pkl_file = os.path.join(
    j.utils.get_assets_dir(), "sample_imgs/strawberry_error.pkl"
)
test_pkl_file = os.path.join(
    j.utils.get_assets_dir(), "sample_imgs/knife_spoon_box_real.pkl"
)
test_pkl_file = os.path.join(j.utils.get_assets_dir(), "sample_imgs/red_lego_multi.pkl")
test_pkl_file = os.path.join(j.utils.get_assets_dir(), "sample_imgs/demo2_nolight.pkl")

file = open(test_pkl_file, "rb")
camera_images = pickle.load(file)["camera_images"]
images = [j.RGBD.construct_from_camera_image(c) for c in camera_images]
rgbd = images[0]

j.get_rgb_image(rgbd.rgb).save("rgb.png")

sam = build_sam(
    checkpoint="/home/nishadgothoskar/jax3dp3/assets/sam/sam_vit_h_4b8939.pth"
)
sam.to(device="cuda")

mask_generator = SamAutomaticMaskGenerator(sam)
boxes = mask_generator.generate(np.array(rgbd.rgb))

full_segmentation = jnp.ones(rgbd.rgb.shape[:2]) * -1.0
num_objects_so_far = 0
for i in range(len(boxes)):
    seg_mask = jnp.array(boxes[i]["segmentation"])

    matched = False
    for jj in range(num_objects_so_far):
        seg_mask_existing_object = full_segmentation == jj

        intersection = seg_mask * seg_mask_existing_object
        if intersection[seg_mask].mean() > 0.9:
            matched = True

    if not matched:
        full_segmentation = full_segmentation.at[seg_mask].set(num_objects_so_far)
        num_objects_so_far += 1

    segmentation_image = j.get_depth_image(
        full_segmentation + 1, max=full_segmentation.max() + 2
    )
    seg_viz = j.get_depth_image(seg_mask)
    j.hstack_images([segmentation_image, seg_viz]).save(f"{i}.png")

full_segmentation = full_segmentation.at[seg_mask].set(i + 1)


# sam = build_sam()
# sam.to(device="cuda")
# mask_generator = SamAutomaticMaskGenerator(sam)

# j.get_rgb_image(rgbd.rgb).save("rgb.png")
# mask_generator.generate(np.array(rgbd.rgb))

# mask_generator.generate(np.array(img))

# sam = sam_model_registry["default"](checkpoint=args.checkpoint)
# _ = sam.to(device=args.device)
# output_mode = "coco_rle" if args.convert_to_rle else "binary_mask"
# amg_kwargs = get_amg_kwargs(args)
# generator = SamAutomaticMaskGenerator(sam, output_mode=output_mode, **amg_kwargs)


# mask_generator = SamAutomaticMaskGenerator["default"](build_sam())
