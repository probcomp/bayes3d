import bayes3d as b
import jax.numpy as jnp

HIINTERFACE = None


def carvekit_get_foreground_mask(image: b.RGBD):
    global HIINTERFACE
    if HIINTERFACE is None:
        import torch
        from carvekit.api.high import HiInterface

        HIINTERFACE = HiInterface(
            object_type="object",  # Can be "object" or "hairs-like".
            batch_size_seg=5,
            batch_size_matting=1,
            device="cuda" if torch.cuda.is_available() else "cpu",
            seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
            matting_mask_size=2048,
            trimap_prob_threshold=220,  # 231,
            trimap_dilation=15,
            trimap_erosion_iters=20,
            fp16=False,
        )
    imgs = HIINTERFACE([b.get_rgb_image(image.rgb)])
    mask = jnp.array(imgs[0])[..., -1] > 0.5
    return mask
