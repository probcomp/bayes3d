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

h, w = 200,200
height,width = h,w
cx = (w-1)/2
cy = (h-1)/2
fx = 200.0
fy = 200.0
near=0.05
far=10.0

P = projection_matrix(h, w, fx, fy, cx, cy, near, far)
perspective_matrix = tuple(P.T.astype('f4').reshape(-1))



glfw.init()
# Set window hint NOT visible
glfw.window_hint(glfw.VISIBLE, False)
# Create a windowed mode window and its OpenGL context
window = glfw.create_window(width, height, "hidden window", None, None)
# Make the window's context current
glfw.make_context_current(window)

glEnable(GL_DEPTH_TEST)
glClear(GL_DEPTH_BUFFER_BIT)

#Now create the shaders
VERTEX_SHADER = shaders.compileShader("""
#version 440
in vec3 in_vert;
uniform mat4 mvp;
out vec4 color;
void main()
{
    gl_Position = mvp * vec4(in_vert, 1.0);
    color = gl_Position;
}
""", GL_VERTEX_SHADER)

FRAGMENT_SHADER = shaders.compileShader("""
#version 440
out vec4 fragColor;
in vec4 color;
void main()
{
    fragColor = vec4(color);
}
""", GL_FRAGMENT_SHADER)

shader = shaders.compileProgram(VERTEX_SHADER, FRAGMENT_SHADER)

mesh = trimesh.load(os.path.join(jax3dp3.utils.get_assets_dir(),"bunny.obj"))
vertices_orig = np.array(mesh.vertices)
vertices = vertices_orig.copy()
pose = t3d.transform_from_pos(np.array([0.0, 0.0, 4.2]))
vertices = t3d.apply_transform(vertices, pose)
indices = np.array(mesh.faces)


vertices = np.array([
    [0.0,0.0,3.0],[3.0,0.0,3.0],[0.0,2.0,3.0], [-3.0,-3.0,8.0]
    ], dtype='f')
indices = np.array([[0,1,2], [1,2,3]], dtype=np.int32)

vertices = np.array(vertices, dtype='f')
indices = np.array(indices, dtype=np.int32)


vertexPositions = glGenBuffers(1)
glBindBuffer(GL_ARRAY_BUFFER, vertexPositions)
glBufferData(GL_ARRAY_BUFFER, vertices, GL_STATIC_DRAW)

indexPositions = glGenBuffers(1)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexPositions)
glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices, GL_STATIC_DRAW)

glEnableVertexAttribArray(0)
glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, None)


glEnable(GL_DEPTH_TEST)
glClearColor(0, 0, 0, 1)
glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

glUseProgram(shader)

glUniformMatrix4fv(glGetUniformLocation(shader,"mvp"), 1, GL_FALSE, perspective_matrix)

#glDrawArrays(GL_TRIANGLES, 0, 3) #This line still works


glBindBuffer(GL_ARRAY_BUFFER, vertexPositions)
glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexPositions)
glDrawElements(GL_TRIANGLES, indices.shape[0]*3, GL_UNSIGNED_INT, None) #This line does work too!

im = glReadPixels(0, 0, width, height, GL_RGBA, GL_FLOAT)
im = im.reshape(height,width,4)

import jax3dp3.viz
jax3dp3.viz.save_depth_image(im[:,:,2],"test.png", min=-5.0,max=5.0)

from IPython import embed; embed()
