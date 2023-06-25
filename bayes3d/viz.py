from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from PIL import Image
import numpy as np
import bayes3d.utils
import matplotlib.pyplot as plt
import matplotlib
import graphviz
import distinctipy
import jax.numpy as jnp

RED = np.array([1.0, 0.0, 0.0])
GREEN = np.array([0.0, 1.0, 0.0])
BLUE = np.array([0.0, 0.0, 1.0])
BLACK = np.array([0.0, 0.0, 0.0])

def make_gif(images, filename):
    images[0].save(
        fp=filename,
        format="GIF",
        append_images=images,
        save_all=True,
        duration=100,
        loop=0,
    )


def make_gif_from_pil_images(images, filename):
    images[0].save(
        fp=filename,
        format="GIF",
        append_images=images[1:],
        save_all=True,
        duration=100,
        loop=0,
    )

def load_image_from_file(filename):
    return Image(filename)


def get_depth_image(image, min=None, max=None, cmap=None):
    if cmap is None:
        cmap = plt.get_cmap('turbo')
    if min is None:
        min = np.min(image)
    if max is None:
        max = np.max(image)
        
    depth = (image - min) / (max - min + 1e-10)
    depth = np.clip(depth, 0, 1)

    img = Image.fromarray(
        np.rint(cmap(depth) * 255.0).astype(np.int8), mode="RGBA"
    )
    return img

def add_rgba_dimension(particles_rendered):
    if particles_rendered.shape[-1] == 3:
        p = jnp.concatenate([particles_rendered, 255.0 * jnp.ones((*particles_rendered.shape[:2],1))],axis=-1)
        return p
    return particles_rendered

def get_rgb_image(image, max=255.0):
    if image.shape[-1] == 3:
        image_type = "RGB"
    else:
        image_type = "RGBA"

    img = Image.fromarray(
        np.rint(
            image / max * 255.0
        ).astype(np.int8),
        mode=image_type,
    ).convert("RGBA")
    return img

def overlay_image(img_1, img_2, alpha=0.5):
    return Image.blend(img_1, img_2, alpha=alpha)

def resize_image(img, h, w):
    return img.resize((w, h))

def scale_image(img, factor):
    w,h = img.size
    return img.resize((int(w * factor), int(h * factor)))

def vstack_images(images, border = 10):
    max_w = 0
    sum_h = (len(images)-1)*border
    for img in images:
        w,h = img.size
        max_w = max(max_w, w)
        sum_h += h

    full_image = Image.new('RGB', (max_w, sum_h), (255, 255, 255))
    running_h = 0
    for img in images:
        w,h = img.size
        full_image.paste(img, (int(max_w/2 - w/2), running_h))
        running_h += h + border
    return full_image

def hstack_images(images, border = 10):
    max_h = 0
    sum_w = (len(images)-1)*border
    for img in images:
        w,h = img.size
        max_h = max(max_h, h)
        sum_w += w

    full_image = Image.new('RGB', (sum_w, max_h),(255, 255, 255))
    running_w = 0
    for img in images:
        w,h = img.size
        full_image.paste(img, (running_w, int(max_h/2 - h/2)))
        running_w += w + border
    return full_image

def hvstack_images(images, h, w, border=10):
    assert len(images) == h * w

    images_to_vstack = []

    for row_idx in range(h):
        hstacked_row = hstack_images(images[row_idx*w:(row_idx+1)*w])
        images_to_vstack.append(hstacked_row)
    
    return vstack_images(images_to_vstack)


####

def multi_panel(images, labels=None, title=None, bottom_text=None, title_fontsize=40, label_fontsize=30,  bottom_fontsize=20, middle_width=10):
    num_images = len(images)
    w = images[0].width
    h = images[0].height

    sum_of_widths = np.sum([img.width for img in images])

    dst = Image.new(
        "RGBA", (sum_of_widths + (num_images - 1) * middle_width, h), (255, 255, 255, 255)
    )

    drawer = ImageDraw.Draw(dst)
    font_bottom = ImageFont.truetype(os.path.join(bayes3d.utils.get_assets_dir(), "fonts", "IBMPlexSerif-Regular.ttf"), bottom_fontsize)
    font_label = ImageFont.truetype(os.path.join(bayes3d.utils.get_assets_dir(), "fonts", "IBMPlexSerif-Regular.ttf"), label_fontsize)
    font_title = ImageFont.truetype(os.path.join(bayes3d.utils.get_assets_dir(), "fonts", "IBMPlexSerif-Regular.ttf"), title_fontsize)

    bottom_border = 0
    title_border = 0
    label_border = 0
    if bottom_text is not None:
        msg = bottom_text
        _, _, text_w, text_h = drawer.textbbox((0, 0), msg, font=font_bottom)
        bottom_border = text_h
    if title is not None:
        msg = title
        _, _, text_w, text_h = drawer.textbbox((0, 0), msg, font=font_title)
        title_border = text_h
    if labels is not None:
        for msg in labels:
            _, _, text_w, text_h = drawer.textbbox((0, 0), msg, font=font_label)
            label_border = max(text_h, label_border)

    bottom_border += 0 
    title_border += 20
    label_border += 20 

    dst = Image.new(
        "RGBA", (sum_of_widths+ (num_images - 1) * middle_width, h + title_border + label_border + bottom_border), (255, 255, 255, 255)
    )
    drawer = ImageDraw.Draw(dst)

    width_counter = 0
    for (j, img) in enumerate(images):
        dst.paste(
            img,
            (width_counter + j * middle_width, title_border + label_border)
        )
        width_counter += img.width

    if title is not None:
        msg = title
        _, _, text_w, text_h = drawer.textbbox((0, 0), msg, font=font_title)
        drawer.text(((sum_of_widths + (num_images - 1) * middle_width)/2.0 - text_w/2 , title_border/2 - text_h/2), msg, font=font_title, fill="black")


    width_counter = 0
    if labels is not None:
        for (i, msg) in enumerate(labels):
            w = images[i].width
            _, _, text_w, text_h = drawer.textbbox((0, 0), msg, font=font_label)
            drawer.text((width_counter + i * middle_width + w/2 - text_w/2, title_border + label_border/2 - text_h/2), msg, font=font_label, fill="black")
            width_counter += w

    if bottom_text is not None:
        msg = bottom_text
        _, _, text_w, text_h = drawer.textbbox((0, 0), msg, font=font_bottom)
        drawer.text((5,  title_border + label_border + h + 5), msg, font=font_bottom, fill="black")

    return dst


def multi_panel_vertical(images, middle_width=10, title_border=20, fontsize=20):
    num_images = len(images)
    w = images[0].width
    h = images[0].height
    dst = Image.new(
        "RGBA", (w, num_images * h + (num_images - 1) * middle_width + title_border), (255, 255, 255, 255)
    )
    for (j, img) in enumerate(images):
        dst.paste(
            img,
            (0, title_border + j * h + j * middle_width)
        )

    return dst
    
def distinct_colors(num_colors, pastel_factor=0.5):
    return [np.array(i) for i in distinctipy.get_colors(num_colors, pastel_factor=pastel_factor)]

def viz_graph(num_nodes, edges, filename, node_names=None):
    if node_names is None:
        node_names = [str(i) for i in range(num_nodes)]

    g_out = graphviz.Digraph()
    g_out.attr("node", style="filled")
    
    colors = matplotlib.cm.tab20(range(num_nodes))
    colors = distinctipy.get_colors(num_nodes, pastel_factor=0.7)
    for i in range(num_nodes):
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
