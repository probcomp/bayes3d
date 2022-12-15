import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib
# import graphviz
# import distinctipy


def make_gif_from_pil_images(images, filename):
    images[0].save(
        fp=filename,
        format="GIF",
        append_images=images,
        save_all=True,
        duration=100,
        loop=0,
    )


def get_depth_image(image, min=0.0, max=1.0):
    cm = plt.get_cmap('turbo')
    img = Image.fromarray(
        np.rint(cm((np.clip(np.array(image), min, max) - min) / (max - min)) * 255.0).astype(np.int8), mode="RGBA"
    )
    return img

def save_depth_image(image, filename, min=0.0, max=1.0):
    img = get_depth_image(image, min=min, max=max)
    img.save(filename)

def get_rgb_image(image, max_val):
    img = Image.fromarray(
        np.rint(
            image / max_val * 255.0
        ).astype(np.int8),
        mode="RGB",
    )
    return img

def save_rgb_image(image, max_val, filename):
    img = get_rgb_image(image, max_val)
    img.save(filename)

def get_rgba_image(image, max_val):
    img = Image.fromarray(
        np.rint(
            image / max_val * 255.0
        ).astype(np.int8),
        mode="RGBA",
    )
    return img


def save_rgba_image(image, max_val, filename):
    img = get_rgba_image(image, max_val)
    img.save(filename)

####

def multi_panel(images, labels, middle_width=10, top_border=20, fontsize=20):
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

    if labels is not None:
        for (i, msg) in enumerate(labels):
            _, _, text_w, text_h = drawer.textbbox((0, 0), msg, font=font)
            drawer.text((i * w + i * middle_width + w/2 - text_w/2, top_border/2 - text_h/2), msg, font=font, fill="black")
    return dst



# Max Val

def viz_graph(num_nodes, edges, filename, node_names=None):
    if node_names is None:
        node_names = [str(i) for i in range(num_nodes)]

    g_out = graphviz.Digraph()
    g_out.attr("node", style="filled")
    
    colors = matplotlib.cm.tab20(range(num_nodes))
    colors = distinctipy.get_colors(num_nodes, pastel_factor=0.7)
    for i in range(len(edges)):
        g_out.node(str(i), node_names[i], fillcolor=matplotlib.colors.to_hex(colors[i]))

    for (i,j) in edges:
        if i==-1:
            continue
        g_out.edge(str(i),str(j))

    max_width_px = 2000
    max_height_px = 2000
    dpi = 200

    g_out.attr("graph",
                # See https://graphviz.gitlab.io/_pages/doc/info/attrs.html#a:size
                size="{},{}!".format(max_width_px / dpi, max_height_px / dpi),
                dpi=str(dpi))
    filename_prefix, filetype = filename.split(".")
    g_out.render(filename_prefix, format=filetype)
