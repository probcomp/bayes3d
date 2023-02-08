import torch
from carvekit.api.high import HiInterface
from PIL import Image
import numpy as np
import pickle

# Check doc strings for more information
interface = HiInterface(object_type="object",  # Can be "object" or "hairs-like".
                        batch_size_seg=5,
                        batch_size_matting=1,
                        device='cuda' if torch.cuda.is_available() else 'cpu',
                        seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
                        matting_mask_size=2048,
                        trimap_prob_threshold=220,#231,
                        trimap_dilation=15,
                        trimap_erosion_iters=20,
                        fp16=False)


for filename in ["strawberry_error-0-color", "knife_sim-0-color", "demo2_nolight-0-color"]:
    images_without_background = interface([f'./tests/data/{filename}.png'])
    cat_wo_bg = images_without_background[0]
    cat_wo_bg.save(f'./tests/out/{filename}_no_background.png')


    img_arr = np.array(cat_wo_bg)
    mask = img_arr[:,:,3]
    mask[mask > 0] = 255
    
    mask = Image.fromarray(mask)

    mask.save(f'./tests/out/{filename}_mask.png')

    with open(f"./tests/out/{filename}_mask.pik", 'wb') as p:
        pickle.dump({'mask': mask}, p)

    from IPython import embed; embed()