import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def make_gif_from_pil_images(images, filename):
    images[0].save(
        fp=filename,
        format="GIF",
        append_images=images,
        save_all=True,
        duration=100,
        loop=0,
    )