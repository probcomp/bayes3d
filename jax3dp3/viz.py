from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
from PIL import Image
import numpy as np
import jax3dp3.utils
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
        append_images=images,
        save_all=True,
        duration=100,
        loop=0,
    )

def get_depth_image(image, min=0.0, max=1.0, cmap=None):
    if cmap is None:
        cmap = plt.get_cmap('turbo')
    img = Image.fromarray(
        np.rint(cmap((np.clip(np.array(image), min, max) - min) / (max - min)) * 255.0).astype(np.int8), mode="RGBA"
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
    font_bottom = ImageFont.truetype(os.path.join(jax3dp3.utils.get_assets_dir(), "fonts", "DMSans-Regular.ttf"), bottom_fontsize)
    font_label = ImageFont.truetype(os.path.join(jax3dp3.utils.get_assets_dir(), "fonts", "DMSans-Regular.ttf"), label_fontsize)
    font_title = ImageFont.truetype(os.path.join(jax3dp3.utils.get_assets_dir(), "fonts", "DMSans-Regular.ttf"), title_fontsize)


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

    bottom_border += 10 
    title_border += 50 
    label_border += 50 


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




def proj_2d(coord_3d, K):
    R = jnp.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])  # camera extrinsics
    img_2d = K @ R @ coord_3d 

    return img_2d / img_2d[-1]


def overlay_bounding_box(tx, ty, tz, grid_width, depth_img, K, save=True):
    pose = np.array([
                [1.        , 0.        , 0.        , tx],
                [0.        , 1.        , 0.        , ty],
                [0.        , 0.        , 1.        , tz],
                [0.        , 0.        , 0.        , 1.]])
    
    minx = miny = minz = -grid_width 
    maxx = maxy = maxz = grid_width

    # 4-----5
    # |\    |\
    # | 0-----1
    # 6-|---7 |
    #  \|    \|
    #   2-----3


    translation_deltas = np.array([[minx, maxy, maxz, 1], # 0
                        [maxx, maxy, maxz, 1],  # 1                   
                        [minx, miny, maxz, 1], # 2
                        [maxx, miny, maxz, 1], # 3
                        [minx, maxy, minz, 1], # 4
                        [maxx, maxy, minz, 1], # 5
                        [minx, miny, minz, 1], # 6
                        [maxx, miny, minz, 1] # 7

                        ]) 
    translated_vertices = (pose @ translation_deltas.T).T 
    vertices_2d = np.array([proj_2d(v, K) for v in translated_vertices])  

    front_maxx, front_maxy = max(vertices_2d[:4, 0]), max(vertices_2d[:4, 1])
    front_minx, front_miny = min(vertices_2d[:4, 0]), min(vertices_2d[:4, 1])

    back_maxx, back_maxy = max(vertices_2d[4:, 0]), max(vertices_2d[4:, 1])
    back_minx, back_miny = min(vertices_2d[4:, 0]), min(vertices_2d[4:, 1])

    # draw front/back planes
    bbox_color = 'red'
    depth_img_draw = ImageDraw.Draw(depth_img)

    depth_img_draw.rectangle([(front_minx, front_miny), (front_maxx, front_maxy)], fill=None, outline=bbox_color, width=1)
    depth_img_draw.rectangle([(back_minx, back_miny), (back_maxx, back_maxy)], fill=None, outline=bbox_color, width=1)

    # draw parallel lines connecting planes
    depth_img_draw.line([(front_maxx, front_miny), (back_maxx, back_miny)], fill=bbox_color, width=1)
    depth_img_draw.line([(front_maxx, front_maxy), (back_maxx, back_maxy)], fill=bbox_color, width=1)

    depth_img_draw.line([(front_minx, front_miny), (back_minx, back_miny)], fill=bbox_color, width=1)
    depth_img_draw.line([(front_minx, front_maxy), (back_minx, back_maxy)], fill=bbox_color, width=1)


    # axes
    # origin = proj_2d(np.array([0,0,0,1]), K)
    # xdir = proj_2d(np.array([5,0,0,1]), K)
    # ydir = proj_2d(np.array([0,5,0,1]), K)
    # zdir = proj_2d(np.array([0,0,5,1]), K)
    # depth_img_draw.line([(origin[0], origin[1]), (xdir[0], xdir[1])], fill='gray', width=1)
    # depth_img_draw.line([(origin[0], origin[1]), (ydir[0], ydir[1])], fill='gray', width=1)
    # depth_img_draw.line([(origin[0], origin[1]), (zdir[0], zdir[1])], fill='gray', width=1)


    if save:
        depth_img.save("bbox.png")

    return depth_img
