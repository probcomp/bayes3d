import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import matplotlib
import jax3dp3.utils
# import graphviz
# import distinctipy

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
    ).convert("RGBA")
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

def overlay_image(img_1, img_2, alpha=0.5):
    return Image.blend(img_1, img_2, alpha=alpha)

def resize_image(img, h, w):
    return img.resize((w, h))

####

def multi_panel(images, labels=None, middle_width=10, top_border=20, fontsize=20, bottom_text=None, bottom_fontsize=10):
    num_images = len(images)
    w = images[0].width
    h = images[0].height

    dst = Image.new(
        "RGBA", (num_images * w + (num_images - 1) * middle_width, h + top_border), (255, 255, 255, 255)
    )

    drawer = ImageDraw.Draw(dst)
    font_bottom = ImageFont.truetype(os.path.join(jax3dp3.utils.get_assets_dir(), "fonts", "DMSans-Regular.ttf"), bottom_fontsize)
    font = ImageFont.truetype(os.path.join(jax3dp3.utils.get_assets_dir(), "fonts", "DMSans-Regular.ttf"), fontsize)

    bottom_border = 0
    if bottom_text is not None:
        msg = bottom_text
        _, _, text_w, text_h = drawer.textbbox((0, 0), msg, font=font_bottom)
        bottom_border = text_h

    dst = Image.new(
        "RGBA", (num_images * w + (num_images - 1) * middle_width, h + top_border + bottom_border + 10), (255, 255, 255, 255)
    )
    drawer = ImageDraw.Draw(dst)

    for (j, img) in enumerate(images):
        dst.paste(
            img,
            (j * w + j * middle_width, top_border)
        )



    if labels is not None:
        for (i, msg) in enumerate(labels):
            _, _, text_w, text_h = drawer.textbbox((0, 0), msg, font=font)
            drawer.text((i * w + i * middle_width + w/2 - text_w/2, top_border/2 - text_h/2), msg, font=font, fill="black")

    if bottom_text is not None:
        msg = bottom_text
        _, _, text_w, text_h = drawer.textbbox((0, 0), msg, font=font_bottom)
        drawer.text(((num_images * w + (num_images - 1) * middle_width)/2.0 - text_w/2,  top_border + h + 5), msg, font=font_bottom, fill="black")
    return dst


def multi_panel_vertical(images, middle_width=10, top_border=20, fontsize=20):
    num_images = len(images)
    w = images[0].width
    h = images[0].height
    dst = Image.new(
        "RGBA", (w, num_images * h + (num_images - 1) * middle_width + top_border), (255, 255, 255, 255)
    )
    for (j, img) in enumerate(images):
        dst.paste(
            img,
            (0, top_border + j * h + j * middle_width)
        )

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
