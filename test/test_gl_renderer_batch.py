
import moderngl
import numpy as np
import jax3dp3.utils
import jax.numpy as jnp
from jax3dp3.viz import save_depth_image
import os
import trimesh
import jax3dp3.transforms_3d as t3d

height, width = 200, 200
cx = (width-1)/2
cy = (height-1)/2
fx = 200.0
fy = 200.0
near,far = 0.001, 100.0
device_idx=0

def orthographic_matrix(left, right, bottom, top, near, far):
    return np.array(
        (
            (2 / (right - left), 0, 0, -(right + left) / (right - left)),
            (0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)),
            (0, 0, -2 / (far - near), -(far + near) / (far - near)),
            (0, 0, 0, 1),
        )
    )

def projection_matrix(w, h, fx, fy, cx, cy, near, far):
    # transform from cv2 camera coordinates to opengl (flipping sign of y and z)
    view = np.eye(4)
    view[1:3] *= -1

    # see http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
    persp = np.zeros((4, 4))
    persp[0, 0] = fx
    persp[1, 1] = fy
    persp[0, 2] = cx
    persp[1, 2] = cy
    persp[2, 2:] = near + far, near * far
    persp[3, 2] = -1
    # transform the camera matrix from cv2 to opengl as well (flipping sign of y and z)
    persp[:2, 1:3] *= -1

    # The origin of the image is in the *center* of the top left pixel.
    # The orthographic matrix should map the whole image *area* into the opengl NDC, therefore the -.5 below:
    orth = orthographic_matrix(-0.5, w - 0.5, -0.5, h - 0.5, near, far)
    return orth @ persp @ view


ctx = moderngl.create_context(
    standalone=True, backend='egl', device_index=device_idx, require=430
    
)
ctx.disable(moderngl.CULL_FACE)
ctx.enable(moderngl.DEPTH_TEST)
fbo = ctx.framebuffer(
    [
        ctx.renderbuffer(
            (width, height), components=4, dtype='f4'
        ),
        ctx.renderbuffer(
            (width, height), components=4, dtype='f4'
        ),
        ctx.renderbuffer(
            (width, height), components=1, dtype='f4'
        ),
    ],
    ctx.depth_renderbuffer((width, height)),
)
projection_matrix = projection_matrix(
    width,
    height,
    fx,
    fy,
    cx,
    cy,
    near,
    far,
)
projection_matrix = tuple(projection_matrix.T.astype('f4').reshape(-1))
camera_pose_mat = tuple((np.eye(4)).T.astype('f4').reshape(-1))

prog = ctx.program(
    vertex_shader="""
        #version 410
        uniform mat4 mvp;
        uniform mat4 pose_mat;
        uniform mat4 camera_pose_mat;
        uniform float obj_id;
        in vec3 in_vert;
        out vec3 color1;
        out vec3 color2;
        out float out_obj_id;
        void main() {
            vec4 point_on_obj = camera_pose_mat * pose_mat * vec4(in_vert, 1);
            gl_Position = mvp * point_on_obj;
            color2 = vec3(point_on_obj);
            out_obj_id = obj_id;
        }
        """,
    fragment_shader="""
        #version 410
        layout(location = 0) out vec4 fragColor2;
        layout(location = 1) out vec4 fragColor;
        layout(location = 2) out float out_out_obj_id;
        in float out_obj_id;
        in vec3 color1;
        in vec3 color2;
        void main() {
            fragColor = vec4(color1, 1.0);
            fragColor2 = vec4(color2, 1.0);
            out_out_obj_id = out_obj_id;
        }
        """,
)
vaos = []

bunny_path = os.path.join(jax3dp3.utils.get_assets_dir(),"bunny.obj")
mesh = trimesh.load(bunny_path)
vertices = np.array(jnp.array((mesh.vertices)[mesh.faces]))
pose = t3d.transform_from_pos(jnp.array([0.0, 0.0, 4.0]))

vao = ctx.simple_vertex_array(
    prog, ctx.buffer(vertices), 'in_vert'
)
vaos.append(vao)

object_indices = [0]
object_transformation_matrices = [pose]
valid_entries = [
    ii
    for ii in range(len(object_indices))
    if object_transformation_matrices[ii] is not None
]

fbo.use()
ctx.clear()
for ii in valid_entries:
    obj_idx = object_indices[ii]
    obj_transform_matrix = object_transformation_matrices[ii]
    prog['mvp'].value = projection_matrix
    prog['camera_pose_mat'].value = camera_pose_mat
    prog['pose_mat'].value = tuple(
        obj_transform_matrix.T.astype('f4').reshape(-1)
    )
    prog['obj_id'].value = obj_idx + 1.0
    vaos[obj_idx].render(mode=moderngl.TRIANGLES)

# cloud_xyz = np.frombuffer(fbo.read(components=4, dtype='f4'), 'f4').reshape((height, width, 4))
point_cloud_image = np.frombuffer(
    fbo.read(attachment=0, components=4, dtype='f4'), 'f4'
).reshape((height, width, 4))
save_depth_image(point_cloud_image[:,:,2], "bunny.png", max=6.0)

from IPython import embed; embed()