from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np


def enumeration_range_bbox_viz(image, minx, maxx, miny, maxy, minz, maxz, fx_fy, cx_cy, filename):
    # Annotate image with enumeration range bbox (in 2D frame)
    # and return the original and annotated images
    f_x, f_y = fx_fy
    c_x, c_y = cx_cy 

    bbox_coords_3d = [(x, y, z) for x in {float(minx), float(maxx)} for y in {float(miny), float(maxy)} for z in {float(minz), float(maxz)}]

    min_u, min_v = np.Inf, np.Inf
    max_u, max_v = np.NINF, np.NINF

    for (x, y, z) in bbox_coords_3d:
        u = ( x / z ) * f_x + c_x 
        v = ( y / z ) * f_y + c_y
        min_u, min_v = np.min([min_u, u]), np.min([min_v, v])
        max_u, max_v = np.max([max_u, u]), np.max([max_v, v])

    anno_img = ImageDraw.Draw(image)
    anno_img.rectangle([(min_u, min_v), (max_u, max_v)], fill=None, outline='red', width=1)
    image.save(filename)

    return 1