import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

def save_depth_image(image, max_depth, filename):
    cm = plt.get_cmap('turbo')
    img = Image.fromarray(
        np.rint(cm(np.array(image) / max_depth) * 255.0).astype(np.int8), mode="RGBA"
    )
    img.save(filename)

def get_depth_image(image, max_depth):
    cm = plt.get_cmap('turbo')
    img = Image.fromarray(
        np.rint(cm(np.array(image) / max_depth) * 255.0).astype(np.int8), mode="RGBA"
    )
    return img

def save_rgb_image(image, max_val, filename):
    img = Image.fromarray(
        np.rint(
            image / max_val * 255.0
        ).astype(np.int8),
        mode="RGB",
    )
    img.save(filename)

def save_rgba_image(image, max_val, filename):
    img = Image.fromarray(
        np.rint(
            image / max_val * 255.0
        ).astype(np.int8),
        mode="RGBA",
    )
    img.save(filename)
    

def save_rgb_image(image, max_val, filename):
    img = Image.fromarray(
        np.rint(
            image / max_val * 255.0
        ).astype(np.int8),
        mode="RGB",
    )
    img.save(filename)

def save_rgba_image(image, max_val, filename):
    img = Image.fromarray(
        np.rint(
            image / max_val * 255.0
        ).astype(np.int8),
        mode="RGBA",
    )
    img.save(filename)


####

def multi_panel(images, labels, middle_width, top_border, fontsize):
    num_images = len(images)
    w = images[0].width
    h = images[0].height
    dst = Image.new(
        "RGBA", (num_images * w + (num_images - 1) * middle_width, h + top_border), (255, 255, 255, 255)
    )
    for (j, img) in enumerate(images):
        dst.paste(
            img,
            (j * w + j * middle_width, top_border)
        )


    drawer = ImageDraw.Draw(dst)
    font = ImageFont.truetype(os.path.join(os.path.dirname(__file__), "fonts", "DMSans-Regular.ttf"), fontsize)

    for (i, msg) in enumerate(labels):
        text_w, text_h = font.getsize(msg)
        drawer.text((i * w + i * middle_width + w/2 - text_w/2, top_border/2 - text_h/2), msg, font=font, fill="black")
    return dst

