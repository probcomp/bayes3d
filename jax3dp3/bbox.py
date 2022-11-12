import jax.numpy as jnp
from jax3dp3.transforms_3d import quaternion_to_rotation_matrix, transform_from_pos
import numpy as np
from PIL import Image, ImageDraw
from jax3dp3.viz.img import get_depth_image

def axis_aligned_bounding_box(object_points):
    maxs = jnp.max(object_points,axis=0)
    mins = jnp.min(object_points,axis=0)
    dims = (maxs - mins)
    center = (maxs + mins) / 2
    return dims, transform_from_pos(center)    

def proj_2d(coord_3d, f_x, f_y, c_x, c_y):
    x,y,z,_ = coord_3d  
    return  ( y / (z + 1e-6) ) * f_y + c_y, ( x / (z + 1e-6) ) * f_x + c_x  # TODO fix this

def overlay_bounding_box(tx, ty, tz, grid_width, depth_img, f_x, f_y, c_x, c_y, save=True):
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

    vertices_frontface = np.array([proj_2d(v, f_x, f_y, c_x, c_y) for v in translated_vertices[[0,1,2,3]]])  
    front_maxx, front_maxy = max(vertices_frontface[:, 0]), max(vertices_frontface[:, 1])
    front_minx, front_miny = min(vertices_frontface[:, 0]), min(vertices_frontface[:, 1])


    vertices_backface = np.array([proj_2d(v, f_x, f_y, c_x, c_y) for v in translated_vertices[[4,5,6,7]]])  
    back_maxx, back_maxy = max(vertices_backface[:, 0]), max(vertices_backface[:, 1])
    back_minx, back_miny = min(vertices_backface[:, 0]), min(vertices_backface[:, 1])

    # draw faces
    bbox_color = 'red'
    depth_img_draw = ImageDraw.Draw(depth_img)

    depth_img_draw.rectangle([(front_minx, front_miny), (front_maxx, front_maxy)], fill=None, outline=bbox_color, width=1)
    depth_img_draw.rectangle([(back_minx, back_miny), (back_maxx, back_maxy)], fill=None, outline=bbox_color, width=1)

    depth_img_draw.line([(front_maxx, front_miny), (back_maxx, back_miny)], fill=bbox_color, width=1)
    depth_img_draw.line([(front_maxx, front_maxy), (back_maxx, back_maxy)], fill=bbox_color, width=1)

    depth_img_draw.line([(front_minx, front_miny), (back_minx, back_miny)], fill=bbox_color, width=1)
    depth_img_draw.line([(front_minx, front_maxy), (back_minx, back_maxy)], fill=bbox_color, width=1)


    if save:
        depth_img.save("bbox.png")

    return depth_img

