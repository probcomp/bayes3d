import jax.numpy as jnp
from jax3dp3.transforms_3d import transform_from_pos
import numpy as np
from PIL import ImageDraw

def axis_aligned_bounding_box(object_points):
    maxs = jnp.max(object_points,axis=0)
    mins = jnp.min(object_points,axis=0)
    dims = (maxs - mins)
    center = (maxs + mins) / 2
    return dims, transform_from_pos(center)    

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
