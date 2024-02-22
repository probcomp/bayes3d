import argparse
import os
import time
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import torch
import nvdiffrast 
import bayes3d as b
import nvdiffrast.torch as dr  # modified nvdiffrast to expose backward fn call api
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import pytorch3d.transforms

# --------------------
# transform points op
# --------------------
def xfm_points(points, matrix):
    """Transform points.
    Args:
        points: Tensor containing 3D points with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
        use_python: Use PyTorch's torch.matmul (for validation)
    Returns:
        Transformed points in homogeneous 4D with shape [minibatch_size, num_vertices, 4].
    """
    out = torch.matmul(
        torch.nn.functional.pad(points, pad=(0, 1), mode="constant", value=1.0),
        torch.transpose(matrix, 1, 2),
    )
    return out


def posevec_to_matrix(position, quat):
    return torch.cat(
        (
            torch.cat((pytorch3d.transforms.quaternion_to_matrix(quat), position.unsqueeze(1)), 1),
            torch.tensor([[0.0, 0.0, 0.0, 1.0]],device=device),
        ),
        0,
    )

intrinsics = b.Intrinsics(
    height=100,
    width=100,
    fx=200.0, fy=200.0,
    cx=50., cy=50.,
    near=0.001, far=16.0
)
proj_cam = torch.from_numpy(
    np.array(
        b.camera._open_gl_projection_matrix(
            intrinsics.height,
            intrinsics.width,
            intrinsics.fx,
            intrinsics.fy,
            intrinsics.cx,
            intrinsics.cy,
            intrinsics.near,
            intrinsics.far,
        )
    )
).cuda()


torch_glctx = dr.RasterizeGLContext()

import trimesh
meshes = []

path = os.path.join(b.utils.get_assets_dir(), "sample_objs/bunny.obj")
bunny_mesh = trimesh.load(path)
bunny_mesh.vertices  = bunny_mesh.vertices * jnp.array([1.0, -1.0, 1.0]) + jnp.array([0.0, 1.0, 0.0])
meshes.append(bunny_mesh)

m = meshes[0]
vertices = torch.from_numpy(m.vertices.astype(np.float32)).cuda()[None,...].contiguous()
faces = torch.from_numpy(m.faces.astype(np.int32)).cuda().contiguous()

pos_gt, quat_gt = (
    torch.from_numpy(np.array([0.0, 0.0, 7.0]).astype(np.float32)).cuda(),
    torch.from_numpy(np.array([0.2, 0.4, 1.0, 0.1]).astype(np.float32)).cuda()
)


def render(pos, quat, stop_grad=False):
    vertices_camera = xfm_points(vertices, posevec_to_matrix(pos, quat)[None,...])
    vertices_clip = xfm_points(vertices_camera[...,:3], proj_cam[None,...]).contiguous()
    rast_out, _ = dr.rasterize(torch_glctx, vertices_clip, faces, resolution=[intrinsics.height, intrinsics.width])
    if stop_grad:
        rast_out = rast_out.detach()
    color   , _ = dr.interpolate(vertices_camera[0,...,2:3].contiguous(), rast_out, faces)
    return color

gt_color = render(pos_gt, quat_gt)


init_pos, init_quat = (
    torch.from_numpy(np.array([-0.3, 0.1, 6.5]).astype(np.float32)).cuda(),
    torch.from_numpy(np.array([0.5, 0.4, 1.0, 0.1]).astype(np.float32)).cuda()
)
pos,quat = init_pos.clone(), init_quat.clone()
pos.requires_grad = True
quat.requires_grad = True


for _ in range(500):
    color = render(pos, quat, stop_grad=False)
    mask = (color[...,-1] > 0.0) * (gt_color[...,-1] > 0.0)
    loss = (torch.abs(gt_color - color) * mask[...,None]).mean()
    loss.backward()
    print(loss)
    pos.data -= 0.1 * pos.grad
    quat.data -= 0.1 * quat.grad
    pos.grad.zero_()
    quat.grad.zero_()
print(quat_gt / torch.linalg.norm(quat_gt))
print(quat / torch.linalg.norm(quat))


b.hstack_images(
    [
        b.get_depth_image(gt_color[0,...,0].detach().cpu().numpy()),
        b.get_depth_image(render(init_pos, init_quat)[0,...,0].detach().cpu().numpy()),
        b.get_depth_image(render(pos, quat)[0,...,0].detach().cpu().numpy())
    ]
).save("depth.png")




from IPython import embed; embed()