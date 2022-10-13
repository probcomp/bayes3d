import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def make_gif(images, max_depth, filename):

    cm = plt.get_cmap('turbo')
    max_depth = 3.0
    image_list = []

    for i in range(images.shape[0]):
        image_list.append(
            Image.fromarray(
                np.rint(cm(np.array(images[i, :, :, 2]) / max_depth) * 255.0).astype(np.int8), mode="RGBA"
            )
        )

    image_list[0].save(
        fp=filename,
        format="GIF",
        append_images=image_list,
        save_all=True,
        duration=100,
        loop=0,
    )


