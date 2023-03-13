from jax3dp3.dino import Dino
import jax3dp3 as j
import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

bop_ycb_dir = os.path.join(j.utils.get_assets_dir(), "bop/ycbv")
rgbd, gt_ids, gt_poses, masks = j.ycb_loader.get_test_img('49', '1', bop_ycb_dir)

model = Dino(rgbd.rgb.shape[0], rgbd.rgb.shape[1])
embeddings = model.get_embeddings(rgbd.rgb)


rgbd2, gt_ids, gt_poses, masks = j.ycb_loader.get_test_img('51', '1', bop_ycb_dir)
embeddings2 = model.get_embeddings(rgbd2.rgb)


clickx, clicky = 320, 180
selected_embedding = embeddings[0, :, clicky, clickx]  # (512,)

def get_similarity( selected_embedding, embeddings, similarity_thresh=0.1):
    similarity = ((selected_embedding.reshape(1,-1,1,1) * embeddings).sum(1) + 1.0) / 2.0
    print(similarity.min(),similarity.max())
    return similarity[0]
    similarity_thresh = 0.1
    similarity_rel = (similarity - similarity.min()) / (
        similarity.max() - similarity.min() + 1e-12
    )
    similarity_rel = similarity_rel[0]  # 1, H // 2, W // 2 -> # H // 2, W // 2
    similarity_rel[similarity_rel < similarity_thresh] = 0.0
    return similarity_rel

similarity_rel1 = get_similarity(selected_embedding, embeddings)
similarity_rel2 = get_similarity(selected_embedding, embeddings2)

cmap = matplotlib.cm.get_cmap("jet")
similarity_colormap1 = j.get_rgb_image(cmap(similarity_rel1)[..., :3] * 255.0)
similarity_colormap2 = j.get_rgb_image(cmap(similarity_rel2)[..., :3] * 255.0)
similarity_colormap1.save("1.png")
similarity_colormap2.save("2.png")

rgb1 = j.get_rgb_image(rgbd.rgb)
overlay = Image.new('RGBA', rgb1.size)
draw = ImageDraw.Draw(overlay)
draw.ellipse([clickx-20, clicky-20, clickx+20, clicky+20],fill=(255, 0, 0, 200))
rgb1.paste(overlay, (0,0), overlay)
rgb1.save("1.png")

rgb2 = j.get_rgb_image(rgbd2.rgb)

j.hstack_images(
    [
        rgb1,
        j.overlay_image(rgb2, similarity_colormap2),
        similarity_colormap2
    ]
).save("1.png")
