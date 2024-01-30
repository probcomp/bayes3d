import functools
import math

import jax
import jax.numpy as jnp
import numpy as np
import torch
from diff_gaussian_rasterization import (
    _C,
    GaussianRasterizationSettings,
    GaussianRasterizer,
)

import bayes3d as b

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def intrinsics_to_rasterizer(intrinsics, camera_pose_jax):
    fovX = jnp.arctan(intrinsics.width / 2 / intrinsics.fx) * 2.0
    fovY = jnp.arctan(intrinsics.height / 2 / intrinsics.fy) * 2.0
    tan_fovx = math.tan(fovX)
    tan_fovy = math.tan(fovY)

    proj_matrix = getProjectionMatrix(0.01, 100.0, fovX, fovY).transpose(0, 1).cuda()
    view_matrix = torch.transpose(
        torch.tensor(np.array(b.inverse_pose(camera_pose_jax))), 0, 1
    ).cuda()

    raster_settings = GaussianRasterizationSettings(
        image_height=int(intrinsics.height),
        image_width=int(intrinsics.width),
        tanfovx=tan_fovx,
        tanfovy=tan_fovy,
        bg=torch.tensor([0.0, 0.0, 0.0]).cuda(),
        scale_modifier=1.0,
        viewmatrix=view_matrix,
        projmatrix=view_matrix @ proj_matrix,
        sh_degree=0,
        campos=torch.zeros(3).cuda(),
        prefiltered=False,
        debug=None,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    return rasterizer


def gaussian_raster_fwd(
    means3D, colors_precomp, opacity, scales, rotations, camera_pose, intrinsics
):
    (
        means3D_torch,
        colors_precomp_torch,
        opacity_torch,
        scales_torch,
        rotations_torch,
        camera_pose_torch,
    ) = [
        b.utils.jax_to_torch(x)
        for x in [means3D, colors_precomp, opacity, scales, rotations, camera_pose]
    ]

    fovX = jnp.arctan(intrinsics.width / 2 / intrinsics.fx) * 2.0
    fovY = jnp.arctan(intrinsics.height / 2 / intrinsics.fy) * 2.0
    tan_fovx = math.tan(fovX)
    tan_fovy = math.tan(fovY)

    proj_matrix = getProjectionMatrix(0.01, 100.0, fovX, fovY).transpose(0, 1).cuda()
    view_matrix = torch.transpose(torch.linalg.inv(camera_pose_torch), 0, 1).cuda()
    raster_settings = GaussianRasterizationSettings(
        image_height=int(intrinsics.height),
        image_width=int(intrinsics.width),
        tanfovx=tan_fovx,
        tanfovy=tan_fovy,
        bg=torch.tensor([0.0, 0.0, 0.0]).cuda(),
        scale_modifier=1.0,
        viewmatrix=view_matrix,
        projmatrix=view_matrix @ proj_matrix,
        sh_degree=1,
        campos=torch.zeros(3).cuda(),
        prefiltered=False,
        debug=None,
    )
    cov3Ds_precomp = torch.Tensor([])
    sh = torch.Tensor([])
    args = (
        raster_settings.bg,
        means3D_torch,
        colors_precomp_torch,
        opacity_torch,
        scales_torch,
        rotations_torch,
        raster_settings.scale_modifier,
        cov3Ds_precomp,
        raster_settings.viewmatrix,
        raster_settings.projmatrix,
        raster_settings.tanfovx,
        raster_settings.tanfovy,
        raster_settings.image_height,
        raster_settings.image_width,
        sh,
        raster_settings.sh_degree,
        raster_settings.campos,
        raster_settings.prefiltered,
        raster_settings.debug,
    )

    (
        num_rendered,
        color,
        radii,
        geomBuffer,
        binningBuffer,
        imgBuffer,
    ) = _C.rasterize_gaussians(*args)
    return b.utils.torch_to_jax(color), (
        intrinsics,
        num_rendered,
        camera_pose,
        colors_precomp,
        means3D,
        scales,
        rotations,
        opacity,
        *[
            b.utils.torch_to_jax(i)
            for i in [radii, geomBuffer, binningBuffer, imgBuffer]
        ],
    )


def gaussian_raster_bwd(saved_tensors, grad_output):
    (
        intrinsics,
        num_rendered,
        camera_pose,
        colors_precomp,
        means3D,
        scales,
        rotations,
        opacity,
        radii,
        geomBuffer,
        binningBuffer,
        imgBuffer,
    ) = saved_tensors

    (
        means3D_torch,
        colors_precomp_torch,
        opacity_torch,
        scales_torch,
        rotations_torch,
        camera_pose_torch,
    ) = [
        b.utils.jax_to_torch(x)
        for x in [means3D, colors_precomp, opacity, scales, rotations, camera_pose]
    ]

    fovX = jnp.arctan(intrinsics.width / 2 / intrinsics.fx) * 2.0
    fovY = jnp.arctan(intrinsics.height / 2 / intrinsics.fy) * 2.0
    tan_fovx = math.tan(fovX)
    tan_fovy = math.tan(fovY)

    proj_matrix = getProjectionMatrix(0.01, 100.0, fovX, fovY).transpose(0, 1).cuda()
    view_matrix = torch.transpose(torch.linalg.inv(camera_pose_torch), 0, 1).cuda()
    raster_settings = GaussianRasterizationSettings(
        image_height=int(intrinsics.height),
        image_width=int(intrinsics.width),
        tanfovx=tan_fovx,
        tanfovy=tan_fovy,
        bg=torch.tensor([0.0, 0.0, 0.0]).cuda(),
        scale_modifier=1.0,
        viewmatrix=view_matrix,
        projmatrix=view_matrix @ proj_matrix,
        sh_degree=1,
        campos=torch.zeros(3).cuda(),
        prefiltered=False,
        debug=None,
    )

    geomBuffer_torch = b.utils.jax_to_torch(geomBuffer)
    binningBuffer_torch = b.utils.jax_to_torch(binningBuffer)
    imgBuffer_torch = b.utils.jax_to_torch(imgBuffer)
    radii_torch = b.utils.jax_to_torch(radii)

    grad_out_color_torch = b.utils.jax_to_torch(grad_output)

    cov3Ds_precomp = torch.Tensor([])
    sh = torch.Tensor([])
    args = (
        raster_settings.bg,
        means3D_torch,
        radii_torch,
        colors_precomp_torch,
        scales_torch,
        rotations_torch,
        raster_settings.scale_modifier,
        cov3Ds_precomp,
        raster_settings.viewmatrix,
        raster_settings.projmatrix,
        raster_settings.tanfovx,
        raster_settings.tanfovy,
        grad_out_color_torch,
        sh,
        raster_settings.sh_degree,
        raster_settings.campos,
        geomBuffer_torch,
        num_rendered,
        binningBuffer_torch,
        imgBuffer_torch,
        raster_settings.debug,
    )

    (
        grad_means2D,
        grad_colors_precomp,
        grad_opacities,
        grad_means3D,
        grad_cov3Ds_precomp,
        grad_sh,
        grad_scales,
        grad_rotations,
    ) = _C.rasterize_gaussians_backward(*args)

    grad_means3D, grad_colors_precomp, grad_opacities, grad_scales, grad_rotations = [
        b.utils.torch_to_jax(i)
        for i in [
            grad_means3D,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
        ]
    ]

    # input order means3D, colors_precomp, opacities, scales, rotations, camera_pose, intrinsics
    return (
        grad_means3D,
        grad_colors_precomp,
        grad_opacities,
        grad_scales,
        grad_rotations,
        None,
        None,
    )


@functools.partial(jax.custom_vjp)
def gaussian_raster(
    means3D, colors_precomp, opacities, scales, rotations, camera_pose, intrinsics
):
    return gaussian_raster_fwd(
        means3D, colors_precomp, opacities, scales, rotations, camera_pose, intrinsics
    )[0]


gaussian_raster.defvjp(gaussian_raster_fwd, gaussian_raster_bwd)
