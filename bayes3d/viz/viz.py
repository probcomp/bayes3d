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
import copy

RED = np.array([1.0, 0.0, 0.0])
GREEN = np.array([0.0, 1.0, 0.0])
BLUE = np.array([0.0, 0.0, 1.0])
BLACK = np.array([0.0, 0.0, 0.0])

def load_image_from_file(filename):
    """Load an image from a file."""
    return Image.open(filename)

def make_gif_from_pil_images(images, filename):
    """Save a list of PIL images as a GIF.
    
    Args:
        images (list): List of PIL images.
        filename (str): Filename to save GIF to.
    """
    images[0].save(
        fp=filename,
        format="GIF",
        append_images=images,
        save_all=True,
        duration=100,
        loop=0,
    )

def preprocess_for_viz(img):
    depth_np = np.array(img)
    depth_np[depth_np >= depth_np.max()] = np.inf
    return depth_np

cmap  = copy.copy(plt.get_cmap('turbo'))
cmap.set_bad(color=(1.0, 1.0, 1.0, 1.0))

def get_depth_image(image, max=None):
    """Convert a depth image to a PIL image.
    
    Args:
        image (np.ndarray): Depth image. Shape (H, W).
        min (float): Minimum depth value for colormap.
        max (float): Maximum depth value for colormap.
        cmap (matplotlib.colors.Colormap): Colormap to use.
    Returns:
        PIL.Image: Depth image visualized as a PIL image.
    """
    depth = np.array(image)
    if max is None:
        maxim = depth.max()
    else:
        maxim = max
    mask = depth < maxim
    depth[np.logical_not(mask)] = np.nan
    vmin = depth[mask].min()
    vmax = depth[mask].max()
    depth = (depth - vmin) / (vmax - vmin)

    img = Image.fromarray(
        np.rint(cmap(depth) * 255.0).astype(np.int8), mode="RGBA"
    ).convert("RGB")
    return img

def get_rgb_image(image, max=255.0):
    """Convert an RGB image to a PIL image.
    
    Args:
        image (np.ndarray): RGB image. Shape (H, W, 3).
        max (float): Maximum value for colormap.
    Returns:
        PIL.Image: RGB image visualized as a PIL image.
    """
    image = np.clip(image, 0.0, max)
    if image.shape[-1] == 3:
        image_type = "RGB"
    else:
        image_type = "RGBA"

    img = Image.fromarray(
        np.rint(
            image / max * 255.0
        ).astype(np.int8),
        mode=image_type,
    ).convert("RGB")
    return img

saveargs = dict(bbox_inches='tight', pad_inches=0)


def add_depth_image(ax, depth):
    d = ax.imshow(preprocess_for_viz(depth),cmap=cmap)
    ax.axis('off')
    return d

def add_rgb_image(ax, rgb):
    ax.imshow(rgb)
    ax.axis('off')


def viz_depth_image(depth):
    """Convert a depth image to a PIL image.
    
    Args:
        image (np.ndarray): Depth image. Shape (H, W).
        min (float): Minimum depth value for colormap.
        max (float): Maximum depth value for colormap.
        cmap (matplotlib.colors.Colormap): Colormap to use.
    Returns:
        PIL.Image: Depth image visualized as a PIL image.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    add_depth_image(ax, depth)
    return fig

def viz_rgb_image(image):
    """Convert an RGB image to a PIL image.
    
    Args:
        image (np.ndarray): RGB image. Shape (H, W, 3).
        max (float): Maximum value for colormap.
    Returns:
        PIL.Image: RGB image visualized as a PIL image.
    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    add_rgb_image(ax, image)
    return fig

def pil_image_from_matplotlib(fig):
    img = Image.frombytes('RGBA', fig.canvas.get_width_height(),bytes(fig.canvas.buffer_rgba()))
    return img

def add_rgba_dimension(image):
    """Add an alpha channel to a particle image if it doesn't already have one.
    
    Args:
        image (np.ndarray): Particle image. Shape (H, W, 3) or (H, W, 4).
    """
    if image.shape[-1] == 3:
        p = jnp.concatenate([image, 255.0 * jnp.ones((*image.shape[:2],1))],axis=-1)
        return p
    return image

def overlay_image(img_1, img_2, alpha=0.5):
    """Overlay two images.
    
    Args:
        img_1 (PIL.Image): First image.
        img_2 (PIL.Image): Second image.
        alpha (float): Alpha value for blending.
    Returns:
        PIL.Image: Overlayed image.
    """
    return Image.blend(img_1, img_2, alpha=alpha)

def resize_image(img, h, w):
    """Resize an image.

    Args:
        img (PIL.Image): Image to resize.
        h (int): Height of resized image.
        w (int): Width of resized image.
    Returns:
        PIL.Image: Resized image.
    """
    return img.resize((w, h))

def scale_image(img, factor):
    """Scale an image.
    
    Args:
        img (PIL.Image): Image to scale.
        factor (float): Scale factor.
    Returns:
        PIL.Image: Scaled image.
    """
    w,h = img.size
    return img.resize((int(w * factor), int(h * factor)))

def vstack_images(images, border = 10):
    """Stack images vertically.

    Args:
        images (list): List of PIL images.
        border (int): Border between images.
    Returns:
        PIL.Image: Stacked image.
    """
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
    """Stack images horizontally.

    Args:
        images (list): List of PIL images.
        border (int): Border between images.
    Returns:
        PIL.Image: Stacked image.
    """
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
    """Stack images in a grid.

    Args:
        images (list): List of PIL images.
        h (int): Number of rows.
        w (int): Number of columns.
        border (int): Border between images.
    Returns:
        PIL.Image: Stacked image. 
    """
    assert len(images) == h * w

    images_to_vstack = []

    for row_idx in range(h):
        hstacked_row = hstack_images(images[row_idx*w:(row_idx+1)*w])
        images_to_vstack.append(hstacked_row)
    
    return vstack_images(images_to_vstack)

def multi_panel(images, labels=None, title=None, bottom_text=None, title_fontsize=40, label_fontsize=30,  bottom_fontsize=20, middle_width=10):
    """Combine multiple images into a single image.
    
    Args:
        images (list): List of PIL images.
        labels (list): List of labels for each image.
        title (str): Title for image.
        bottom_text (str): Text for bottom of image.
        title_fontsize (int): Font size for title.
        label_fontsize (int): Font size for labels.
        bottom_fontsize (int): Font size for bottom text.
        middle_width (int): Width of border between images.
    Returns:
        PIL.Image: Combined image.
    """
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

def distinct_colors(num_colors, pastel_factor=0.5):
    """Get a list of distinct colors.
    
    Args:
        num_colors (int): Number of colors to generate.
        pastel_factor (float): Pastel factor.
    Returns:
        list: List of colors.
    """
    return [np.array(i) for i in distinctipy.get_colors(num_colors, pastel_factor=pastel_factor)]

def viz_graph(num_nodes, edges, filename, node_names=None):
    """Visualize a graph.
    
    Args:
        num_nodes (int): Number of nodes in graph.
        edges (list): List of edges in graph.
        filename (str): Filename to save graph to.
        node_names (list): List of node names.
    """
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
