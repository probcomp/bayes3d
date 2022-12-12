import os
import numpy as np
import torch
import nvdiffrast.torch as dr

def projection_matrix(h, w, fx, fy, cx, cy, near, far):
    # transform from cv2 camera coordinates to opengl (flipping sign of y and z)
    view = torch.eye(4,device='cuda')
    view[1:3] *= -1

    # see http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
    persp = torch.zeros((4, 4),device='cuda')
    persp[0, 0] = fx
    persp[1, 1] = fy
    persp[0, 2] = cx
    persp[1, 2] = cy
    persp[2, 2] = near + far
    persp[2, 3] = near * far
    persp[3, 2] = -1
    # transform the camera matrix from cv2 to opengl as well (flipping sign of y and z)
    persp[:2, 1:3] *= -1

    # The origin of the image is in the *center* of the top left pixel.
    # The orthographic matrix should map the whole image *area* into the opengl NDC, therefore the -.5 below:

    left, right, bottom, top = -0.5, w - 0.5, -0.5, h - 0.5
    orth = torch.tensor(
        [
            (2 / (right - left), 0, 0, -(right + left) / (right - left)),
            (0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)),
            (0, 0, -2 / (far - near), -(far + near) / (far - near)),
            (0, 0, 0, 1),
        ],device='cuda'
    )
    return orth @ persp @ view


def render_depth(glenv, vertices, triangles, h,w,fx,fy,cx,cy, near, far):
    proj = projection_matrix(h, w, fx, fy, cx, cy, near, far)
    view_space_vertices_h = torch.concatenate([vertices, torch.ones((*vertices.shape[:-1],1) , device='cuda')],axis=-1)
    clip_space_vertices = torch.einsum("ij,abj->abi", proj, view_space_vertices_h).contiguous()
    rast, _ = dr.rasterize(glenv, clip_space_vertices, triangles, resolution=[h,w], grad_db=False)
    # depth = rast[:,:,:,2]
    # neg_mask = depth == 0
    # depth = 2 * near * far / (far + near - depth * (far - near))
    # depth[neg_mask] = 0

    # xs= torch.linspace(0, w - 1, w, device='cuda')
    # ys= torch.linspace(0, h - 1, h, device='cuda')
    # vu = (torch.stack(torch.meshgrid([ys, xs], indexing="ij"),axis=-1) - torch.tensor([cy,cx],device='cuda')) / torch.tensor([fy,fx],device='cuda')
    # vu = torch.concatenate([vu, torch.ones((*vu.shape[:-1],1), device='cuda')],axis=-1)
    # point_cloud = (vu[None,:,:,:] * depth[:,:,:,None])
    return rast