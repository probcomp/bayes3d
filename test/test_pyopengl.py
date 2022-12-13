from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL import shaders
from OpenGL.arrays import vbo

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

height, width = 200, 200
h,w = height, width
cx = (width-1)/2
cy = (height-1)/2
fx = 200.0
fy = 200.0
near = 0.001
far = 100.0

P = projection_matrix(h, w, fx, fy, cx, cy, near, far)
perspective_matrix = tuple(P.T.astype('f4').reshape(-1))

glfw.init()
# Set window hint NOT visible
glfw.window_hint(glfw.VISIBLE, False)
# Create a windowed mode window and its OpenGL context
window = glfw.create_window(width, height, "hidden window", None, None)
# Make the window's context current
glfw.make_context_current(window)

glEnable(GL_DEPTH_TEST)  # https://www.khronos.org/opengles/sdk/docs/man/xhtml/glEnable.xml
glDepthFunc(GL_LESS)
glClearDepth(1.0)
# glClearColor(0., 0., 0., 1.0)
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

fbo = glGenFramebuffers(1)
glBindFramebuffer(GL_FRAMEBUFFER, fbo)

glDrawBuffers(1, [GL_COLOR_ATTACHMENT0])

samples = 10

depth_tex = glGenTextures(1)

color_tex = glGenTextures(1)
glBindTexture(GL_TEXTURE_2D_ARRAY, color_tex)
glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, color_tex, 0)
glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_RGBA32F, width, height, samples, 0,
                GL_RGBA, GL_UNSIGNED_BYTE, None)
glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)

# glBindTexture(GL_TEXTURE_2D_ARRAY, depth_tex)
# glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, depth_tex, 0)
# glTexImage3D.wrappedOperation(
#     GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH24_STENCIL8, width, height, samples, 0, 
#     GL_DEPTH_STENCIL, GL_UNSIGNED_INT_24_8, None
# );



if not glCheckFramebufferStatus(GL_FRAMEBUFFER) == GL_FRAMEBUFFER_COMPLETE:
    print('framebuffer binding failed')
    exit()
glBindFramebuffer(GL_FRAMEBUFFER, 0)



mesh = trimesh.load(os.path.join(jax3dp3.utils.get_assets_dir(),"bunny.obj"))
vertices_orig = np.array(mesh.vertices)
vertices = vertices_orig.copy()
pose = t3d.transform_from_pos(np.array([0.0, 0.0, 2.0]))
vertices = t3d.apply_transform(vertices, pose)
indices = np.array(mesh.faces)


# vertices = np.array([
#     [0.0,0.0,3.0],[3.0,0.0,3.0],[0.0,2.0,3.0], [-3.0,-3.0,8.0]
#     ], dtype='f')
# indices = np.array([[0,1,2], [1,2,3]], dtype=np.int32)

vertices = np.array(vertices, dtype='f')
indices = np.array(indices, dtype=np.int32)



vao = glGenVertexArrays(1)
glBindVertexArray(vao)

vertexPositions = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vertexPositions)
glBufferData(GL_ARRAY_BUFFER, vertices, GL_STATIC_DRAW)
glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, None)
glEnableVertexAttribArray(0)

indexPositions = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexPositions)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices, GL_STATIC_DRAW)

glBindBuffer(GL_ARRAY_BUFFER, 0)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0)
glBindVertexArray(0)



glEnable(GL_DEPTH_TEST)
glClearColor(0, 0, 0, 1)
glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)


#Now create the shaders
VERTEX_SHADER = shaders.compileShader("""
#version 430
#extension GL_ARB_shader_draw_parameters : enable
in vec3 in_vert;
uniform mat4 mvp;
out vec4 color;
out int v_layer;
void main()
{
    gl_Position = mvp * vec4(in_vert, 1.0);
    color = gl_Position;
    v_layer = gl_DrawIDARB;
}
""", GL_VERTEX_SHADER)

GEOMETRY_SHADER = shaders.compileShader("""
#version 430
#extension GL_ARB_shader_draw_parameters : enable
layout (triangles) in;
layout(triangle_strip, max_vertices=3) out;
in vec4 color[];
in int v_layer[];
out vec4 colorz;
void main()
{
    gl_Layer =  v_layer[0];
    gl_Position = gl_in[0].gl_Position;
    colorz = color[0];
    EmitVertex();

    gl_Layer =  v_layer[0];
    gl_Position = gl_in[1].gl_Position;
    colorz = color[1];
    EmitVertex();

    gl_Layer =  v_layer[0];
    gl_Position = gl_in[2].gl_Position;
    colorz = color[2];
    EmitVertex();

	EndPrimitive();

}
""", GL_GEOMETRY_SHADER)

FRAGMENT_SHADER = shaders.compileShader("""
#version 430
#extension GL_ARB_shader_draw_parameters : enable
out vec4 fragColor;
in vec4 colorz;
void main()
{
    fragColor = vec4(colorz);
}
""", GL_FRAGMENT_SHADER)

shader = shaders.compileProgram(VERTEX_SHADER, GEOMETRY_SHADER, FRAGMENT_SHADER)


glUseProgram(shader)

glUniformMatrix4fv(glGetUniformLocation(shader,"mvp"), 1, GL_FALSE, perspective_matrix)

#glDrawArrays(GL_TRIANGLES, 0, 3) #This line still works


glBindFramebuffer(GL_FRAMEBUFFER, fbo)
glBindVertexArray(vao)
glBindBuffer(GL_ARRAY_BUFFER, vertexPositions)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexPositions)

# indirect = np.array([[3, 10, 0, 0], [3, 5, 1, 0]], dtype=np.uint32)
# glMultiDrawArraysIndirect(GL_TRIANGLES, indirect, 2, 16)

# glDrawArraysInstancedBaseInstance(GL_TRIANGLES, 0, indices.shape[0]*3, 1000, 0)


# glDrawElementsInstancedBaseVertexBaseInstance(
#     GL_TRIANGLES,
#     indices.shape[0]*3,
#     GL_UNSIGNED_INT,
#     None,
#     1,
#     0,
#     0
# )


indirect = np.array([
    [indices.shape[0]*3, samples, 0, 0, 0, 1]
    for _ in range(samples)
    ], dtype=np.uint32)
glMultiDrawElementsIndirect(GL_TRIANGLES,
 	GL_UNSIGNED_INT,
 	indirect,
 	samples,
 	indirect.dtype.itemsize * indirect.shape[1]
)


glBindTexture(GL_TEXTURE_2D_ARRAY, color_tex)
im = glGetTexImage(GL_TEXTURE_2D_ARRAY,  0, GL_RGBA, GL_FLOAT);
im2 = im.reshape(samples, height,width, 4)
print(np.where(im2 > 0))
jax3dp3.viz.save_depth_image(im2[0,:,:,2],"bunny2.png", max=6.0)
jax3dp3.viz.save_depth_image(im2[1,:,:,2],"bunny3.png", max=6.0)


# im = glReadPixels(0, 0, width,height, GL_RGBA, GL_FLOAT)
# im = im[:,:,:]
# jax3dp3.viz.save_depth_image(im[:,:,2],"bunny2.png", max=6.0)



from IPython import embed; embed()




