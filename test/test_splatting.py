import diff_gaussian_rasterization as dgr
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import bayes3d as b
import jax.numpy as jnp

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

intrinsics = b.Intrinsics(
    height=100,
    width=100,
    fx=50.0, fy=50.0,
    cx=50.0, cy=50.0,
    near=0.1, far=6.0
)
# proj_matrix = torch.tensor(b.camera._open_gl_projection_matrix(intrinsics.height, intrinsics.width, intrinsics.fx, intrinsics.fy, intrinsics.cx, intrinsics.cy, intrinsics.near, intrinsics.far).astype(np.float32), device=device)
# print(proj_matrix)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

fovX  = jnp.deg2rad(45)
fovY = jnp.deg2rad(45)
proj_matrix = torch.tensor(getProjectionMatrix(intrinsics.near, intrinsics.far, fovX, fovY), device=device)

N = 1
# means3D = torch.rand((N, 3)).cuda() - 0.5 + torch.tensor([0.0, 0.0, 4.8],device= device)
means3D = torch.tensor([[-0.01, 0.01, 1.0]],device= device)
means2D = torch.ones((N, 3),device= device)
colors = torch.rand((N, 3)).cuda()
opacity = torch.rand((N, 1)).cuda() + 0.5
scales = torch.rand((N, 3)).cuda() * 0.1
rotations = torch.rand((N, 4)).cuda()

# tan_fovx = intrinsics.width  / intrinsics.fx / 2.0 
# tan_fovy = intrinsics.hei"ght / intrinsics.fy / 2.0
# print(tan_fovx, tan_fovy)

tan_fovx = math.tan(fovX)
tan_fovy = math.tan(fovY)

raster_settings = GaussianRasterizationSettings(
    image_height=int(intrinsics.height),
    image_width=int(intrinsics.width),
    tanfovx=tan_fovx,
    tanfovy=tan_fovy,
    bg=torch.tensor([1.,1.,1.]).cuda(),
    scale_modifier=1.0,
    viewmatrix=torch.eye(4).cuda(),
    projmatrix=proj_matrix,
    sh_degree=1,
    campos=torch.zeros(3).cuda(),
    prefiltered=False,
    debug=None
)

rasterizer = GaussianRasterizer(raster_settings=raster_settings)


ground_truth_image, radii = rasterizer(
    means3D = means3D,
    means2D = means2D,
    shs = None,
    colors_precomp = colors,
    opacities = opacity,
    scales = scales,
    rotations = rotations,
    cov3D_precomp = None
)


from IPython import embed; embed()

p_hom = torch.transpose(proj_matrix,0,1) @ torch.tensor([means3D[0,0], means3D[0,1],means3D[0,2], 1.0], device= device)
print(p_hom)
p_proj = p_hom / p_hom[3]
print(p_proj)







#   proj_matrix = torch.tensor(getProjectionMatrix(intrinsics.near, intrinsics.far, fovX, fovY), device=device)
# p_orig -0.010000 0.010000 1.000000 
#  p_hom -0.024142 0.024142 2.016949 -0.101695
#  p_proj 0.237398 -0.237398 -19.833355
#  point_image 61.369896 37.630104