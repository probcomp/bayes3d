import moderngl
import numpy as np


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

class GLRenderer:
    def __init__(self, height, width, fx, fy, cx, cy, near, far, device_idx=0):
        self.height, self.width = height, width
        self.fx, self.fy = fx, fy
        self.cx, self.cy = cx, cy
        self.near, self.far = near, far

        self.ctx = moderngl.create_context(
            standalone=True, backend='egl', device_index=device_idx
        )
        self.ctx.disable(moderngl.CULL_FACE)
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.fbo = self.ctx.framebuffer(
            [
                self.ctx.renderbuffer(
                    (self.width, self.height), components=4, dtype='f4'
                ),
                self.ctx.renderbuffer(
                    (self.width, self.height), components=4, dtype='f4'
                ),
                self.ctx.renderbuffer(
                    (self.width, self.height), components=1, dtype='f4'
                ),
            ],
            self.ctx.depth_renderbuffer((self.width, self.height)),
        )
        self.projection_matrix = projection_matrix(
            self.width,
            self.height,
            self.fx,
            self.fy,
            self.cx,
            self.cy,
            self.near,
            self.far,
        )

        self.prog = self.ctx.program(
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
        self.vaos = []
        self.vertices = []
        self.default_camera_pose = tuple((np.eye(4)).T.astype('f4').reshape(-1))

    def load_vertices(self, vertices):
        """load_object_from_path.

        Args:
            path (str): path to the object mesh
            bop_obj_idx (int): Optionally specify the object index in BOP (from 1 to 21)
        """
        vao = self.ctx.simple_vertex_array(
            self.prog, self.ctx.buffer(vertices), 'in_vert'
        )
        self.vaos.append(vao)
        self.vertices.append(vertices)

    def render(self, object_indices, object_transformation_matrices):
        # Support having Nones in object_transformation_matrices
        valid_entries = [
            ii
            for ii in range(len(object_indices))
            if object_transformation_matrices[ii] is not None
        ]
        projection_matrix = tuple(self.projection_matrix.T.astype('f4').reshape(-1))
        camera_pose_mat = self.default_camera_pose

        self.fbo.use()
        self.ctx.clear()
        for ii in valid_entries:
            obj_idx = object_indices[ii]
            obj_transform_matrix = object_transformation_matrices[ii]
            self.prog['mvp'].value = projection_matrix
            self.prog['camera_pose_mat'].value = camera_pose_mat
            self.prog['pose_mat'].value = tuple(
                obj_transform_matrix.T.astype('f4').reshape(-1)
            )
            self.prog['obj_id'].value = obj_idx + 1.0
            self.vaos[obj_idx].render(mode=moderngl.TRIANGLES)

        # cloud_xyz = np.frombuffer(self.fbo.read(components=4, dtype='f4'), 'f4').reshape((self.height, self.width, 4))
        point_cloud_image = np.frombuffer(
            self.fbo.read(attachment=0, components=4, dtype='f4'), 'f4'
        ).reshape((self.height, self.width, 4))
        return point_cloud_image[:, :, :3]