from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL import shaders
from OpenGL.arrays import vbo

import time
import trimesh
import os
import glfw
import numpy as np
import jax3dp3.utils
import jax3dp3
import jax3dp3.transforms_3d as t3d
import jax3dp3.viz


def projection_matrix(h, w, fx, fy, cx, cy, near, far):
    # transform from cv2 camera coordinates to opengl (flipping sign of y and z)
    view = np.eye(4)
    view[1:3] *= -1

    # see http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
    persp = np.zeros((4, 4))
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
    orth = np.array(
        [
            (2 / (right - left), 0, 0, -(right + left) / (right - left)),
            (0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)),
            (0, 0, -2 / (far - near), -(far + near) / (far - near)),
            (0, 0, 0, 1),
        ]
    )
    return orth @ persp @ view

height, width = 120, 160
h,w = height, width
cx = (width-1)/2
cy = (height-1)/2
fx = 200.0
fy = 200.0
near = 0.001
far = 100.0

P = projection_matrix(h, w, fx, fy, cx, cy, near, far)
perspective_matrix = tuple(P.T.astype('f4').reshape(-1))

mesh = trimesh.load(os.path.join(jax3dp3.utils.get_assets_dir(),"bunny.obj"))
vertices_orig = np.array(mesh.vertices)
vertices = vertices_orig.copy()
indices = np.array(mesh.faces)


# vertices = np.array([
#     [0.0,0.0,3.0],[3.0,0.0,3.0],[0.0,2.0,3.0], [-3.0,-3.0,8.0]
#     ], dtype='f')
# indices = np.array([[0,1,2], [1,2,3]], dtype=np.int32)

vertices = np.array(vertices, dtype='f')
# vertices = np.concatenate([vertices, np.ones((vertices.shape[0],1),dtype='f')], axis=-1)
assert vertices.shape[1] == 3
indices = np.array(indices, dtype=np.int32)


glfw.init()
# Set window hint NOT visible
glfw.window_hint(glfw.VISIBLE, False)
# Create a windowed mode window and its OpenGL context
window = glfw.create_window(width, height, "hidden window", None, None)
# Make the window's context current
glfw.make_context_current(window)

fbo = glGenFramebuffers(1)
glBindFramebuffer(GL_FRAMEBUFFER, fbo)

vao = glGenVertexArrays(1)
glBindVertexArray(vao)

vertexPositions = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vertexPositions)
glBufferData(GL_ARRAY_BUFFER, vertices, GL_DYNAMIC_DRAW)
glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, None)
glEnableVertexAttribArray(0)

indexPositions = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexPositions)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices, GL_DYNAMIC_DRAW)

glEnable(GL_DEPTH_TEST)  # https://www.khronos.org/opengles/sdk/docs/man/xhtml/glEnable.xml
glDepthFunc(GL_LESS)
glClearDepth(1.0)

glDrawBuffers(1, [GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1])

batch_size = 1000

color_tex = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D_ARRAY, color_tex)
glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, color_tex, 0)

depth_tex = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D_ARRAY, depth_tex)
glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, depth_tex, 0)


glBindTexture(GL_TEXTURE_2D_ARRAY, color_tex)
glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA32F, width, height, batch_size, 0,
                GL_RGBA, GL_UNSIGNED_BYTE, None)
glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

glBindTexture(GL_TEXTURE_2D_ARRAY, depth_tex)
glTexImage3D.wrappedOperation(
    GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH24_STENCIL8, width, height, batch_size, 0, 
    GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, None
);

# glDrawBuffers(1, [GL_COLOR_ATTACHMENT0])

if not glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE:
    print('framebuffer binding failed')
    exit()
# glBindFramebuffer(GL_FRAMEBUFFER, 0)


#Now create the shaders
VERTEX_SHADER = shaders.compileShader("""
#version 430
#extension GL_ARB_shader_draw_parameters : enable
in vec3 in_vert;
uniform mat4 pose_mat;
uniform mat4 mvp;
out int v_layer;
out vec4 vertex_on_object;
void main()
{
    v_layer = gl_DrawIDARB;
    vertex_on_object = pose_mat * vec4(in_vert, 1.0);
    gl_Position = mvp * vertex_on_object;
}
""", GL_VERTEX_SHADER)

GEOMETRY_SHADER = shaders.compileShader("""
#version 430
#extension GL_ARB_shader_draw_parameters : enable
layout (triangles) in;
layout(triangle_strip, max_vertices=3) out;
in vec4 vertex_on_object[];
in int v_layer[];
out vec4 vertex_on_object_out;
void main()
{
    gl_Layer =  v_layer[0];
    gl_Position = gl_in[0].gl_Position;
    vertex_on_object_out = vertex_on_object[0];
    EmitVertex();
    gl_Layer =  v_layer[0];
    gl_Position = gl_in[1].gl_Position;
    vertex_on_object_out = vertex_on_object[1];
    EmitVertex();
    gl_Layer =  v_layer[0];
    gl_Position = gl_in[2].gl_Position;
    vertex_on_object_out = vertex_on_object[2];
    EmitVertex();
	EndPrimitive();
}
""", GL_GEOMETRY_SHADER)

FRAGMENT_SHADER = shaders.compileShader("""
#version 430
#extension GL_ARB_shader_draw_parameters : enable
in vec4 vertex_on_object_out;
out vec4 fragColor;
void main()
{
    fragColor = vec4(vertex_on_object_out);
}
""", GL_FRAGMENT_SHADER)

shader = shaders.compileProgram(VERTEX_SHADER, GEOMETRY_SHADER, FRAGMENT_SHADER)

glBindTexture(GL_TEXTURE_2D_ARRAY, color_tex)
glFramebufferTexture(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0, color_tex, 0)


glUseProgram(shader)

glViewport(0, 0, width, height);
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT)

glBindVertexArray(vao)
glBindBuffer(GL_ARRAY_BUFFER, vertexPositions)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexPositions)

glBindFramebuffer(GL_FRAMEBUFFER, fbo)
glBindTexture(GL_TEXTURE_2D_ARRAY, color_tex)

pose = np.array(t3d.transform_from_pos(np.array([0.0, 0.0, 4.0])))
pose = tuple(pose.T.astype('f4').reshape(-1))
glUniformMatrix4fv(glGetUniformLocation(shader,"pose_mat"), 1, GL_FALSE, pose)
glUniformMatrix4fv(glGetUniformLocation(shader,"mvp"), 1, GL_FALSE, perspective_matrix)


start = time.time()

indirect = np.array([
    [indices.shape[0]*3, batch_size, 0, 0, 0, 1]
    for _ in range(batch_size)
    ], dtype=np.uint32)
glMultiDrawElementsIndirect(GL_TRIANGLES,
 	GL_UNSIGNED_INT,
 	indirect,
 	batch_size,
 	indirect.dtype.itemsize * indirect.shape[1]
)


glBindTexture(GL_TEXTURE_2D_ARRAY, color_tex)
im = glGetTexImage(GL_TEXTURE_2D_ARRAY,  0, GL_RGBA, GL_FLOAT);
im2 = im.reshape(batch_size, height,width, 4)

end = time.time()
print ("Time elapsed:", end - start)
print ("FPS:", batch_size / (end - start))

print(np.where(im2 > 0))
jax3dp3.viz.save_depth_image(im2[0,:,:,2],"bunny2.png", max=6.0)
jax3dp3.viz.save_depth_image(im2[-1,:,:,2],"bunny3.png", max=6.0)

from IPython import embed; embed()



