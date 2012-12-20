from OpenGL.GL import *
from OpenGL.GL.framebufferobjects import *
from OpenGL.GL import shaders
import numpy as np
import cv
import camera
import glxcontext
from contextlib import contextmanager

if not 'initialized' in globals():
    initialized = False

def initialize():
    global initialized
    glxcontext.makecurrent()

    if initialized: return

    global fbo
    global rb, rbc, rbs

    fbo = glGenFramebuffers(1)
    glBindFramebuffer(GL_FRAMEBUFFER, fbo)

    # Create three render buffers:
    #    1) xyz/depth  2) normals 3)  4) internal depth
    rbXYZD,rbN,rbInds,rbD = glGenRenderbuffers(4)

    # 1) XYZ/D buffer
    glBindRenderbuffer(GL_RENDERBUFFER, rbXYZD)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, 640, 480)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER,
                              GL_COLOR_ATTACHMENT0,
                              GL_RENDERBUFFER, rbXYZD)

    # 2) Normals buffer
    glBindRenderbuffer(GL_RENDERBUFFER, rbN)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA32F, 640, 480)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER,
                              GL_COLOR_ATTACHMENT1,
                              GL_RENDERBUFFER, rbN)

    # 3) Index buffer
    glBindRenderbuffer(GL_RENDERBUFFER, rbInds)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_RGB8, 640, 480)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER,
                              GL_COLOR_ATTACHMENT2,
                              GL_RENDERBUFFER, rbInds)

    # 4) depth buffer
    glBindRenderbuffer(GL_RENDERBUFFER, rbD)
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, 640, 480)
    glFramebufferRenderbuffer(GL_FRAMEBUFFER,
                              GL_DEPTH_ATTACHMENT,
                              GL_RENDERBUFFER, rbD)

    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    global VERTEX_SHADER, FRAGMENT_SHADER, SYNTHSHADER

    # This vertex shader simulates a range image camera. It takes in 
    # camera (kinect) parameters as well as model parameters. It project
    # the model points into the camera space in order to get the right
    # correct geometry. We additionally precompute the world-coordinates
    # XYZ for each pixel, as well as the world-coordinate normal vectors.
    # (This eliminates some of the steps that would normally be computed
    # using image processing, i.e. filtering, subtraction, normals.)
    VERTEX_SHADER = shaders.compileShader("""#version 120
        varying vec4 v_xyz;
        varying vec3 n_xyz;
        varying vec4 position;
        varying vec4 color;

        mat3 mat4tomat3(mat4 m) { return mat3(m[0].xyz, m[1].xyz, m[2].xyz); }

        void main() {
            v_xyz = gl_ModelViewMatrix * gl_Vertex;
            n_xyz = normalize(gl_NormalMatrix * gl_Normal);
            position = gl_ModelViewProjectionMatrix * gl_Vertex;
            gl_Position = position;
            color = gl_Color;
        }""", GL_VERTEX_SHADER)

    
    FRAGMENT_SHADER = shaders.compileShader("""#version 120
        varying vec4 v_xyz;
        varying vec3 n_xyz;
        varying vec4 position;
        varying vec4 color;
        varying out vec4 xyzd;
        varying out vec4 nxyz;
        varying out vec4 inds;

        void main() {
            xyzd.w = 200 / (1.0 - position.z/position.w);
            xyzd.xyz = v_xyz.xyz;
            nxyz.xyz = n_xyz;
            inds = color;
        }""", GL_FRAGMENT_SHADER)

    SYNTHSHADER = shaders.compileProgram(VERTEX_SHADER, FRAGMENT_SHADER)
    initialized = True


@contextmanager
def render(camera, rect=((0,0),(640,480))):
    initialize()
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glDrawBuffers([GL_COLOR_ATTACHMENT0,GL_COLOR_ATTACHMENT1,GL_COLOR_ATTACHMENT2])
    (L,T),(R,B) = rect
    glViewport(0, 0, 640, 480)
    glClearColor(0,0,0,0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    glOrtho(0, 640, 0, 480, -10, 0)
    # Configure the shader
    shaders.glUseProgram(SYNTHSHADER)

    if 1:
        # 
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glMultMatrixf(np.linalg.inv(camera.KK).transpose())
        glMultMatrixf(np.linalg.inv(camera.RT).transpose())
    else:
        # Upload uniform parameters (model matrix)
        def uniform(f, name, mat, n=1):
            assert mat.dtype == np.float32
            mat = np.ascontiguousarray(mat)
            f(glGetUniformLocation(SYNTHSHADER, name), n, True, mat)
        view_inv = np.linalg.inv(np.dot(camera.RT, camera.KK))
        uniform(glUniformMatrix4fv, 'view_inv', view_inv)

    # Return (yield) a function to copy the result GPU->Host 
    def read(debug=False):
        global readpixels, readpixelsA, readpixelsB, depth

        glReadBuffer(GL_COLOR_ATTACHMENT0)
        readpixelsA = glReadPixels(L, T, R-L, B-T, GL_RGBA, GL_FLOAT,
                                   outputType='array').reshape((480,640,4))

        glReadBuffer(GL_COLOR_ATTACHMENT1)
        readpixelsB = glReadPixels(L, T, R-L, B-T, GL_RGB, GL_FLOAT,
                                   outputType='array').reshape((480,640,3))

        glReadBuffer(GL_COLOR_ATTACHMENT2)
        readpixelsC = glReadPixels(L, T, R-L, B-T, GL_RGBA, GL_UNSIGNED_BYTE,
                                   outputType='array').reshape((480,640,4))

        # Returns the distance in milliunits
        depth = readpixelsA[:,:,3]

        # Sanity check
        if debug:
            if depth.max() == 0: print 'Degenerate (empty) depth image'
            print 'Checking two equivalent depth calculations'
            old = np.seterr(divide='ignore')
            within_eps = lambda a,b: np.all(np.abs(a - b) < 2)
            readpixels = glReadPixels(L, T, R-L, B-T, GL_DEPTH_COMPONENT, GL_FLOAT).reshape((480,640))
            depth2 = (100.0 / np.nan_to_num(1.0 - readpixels)).astype('u2')
            assert within_eps(depth2, depth)
            np.seterr(**old)
            print 'OK'

        # Also return the world coordinates and normals
        xyz = np.ascontiguousarray(readpixelsA[:,:,:3])
        normals = readpixelsB[:,:,:3]

        # Finally, return the color image (typically an index)
        inds = readpixelsC
        return depth, xyz, normals, inds

    try:
        yield read

    finally:
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        shaders.glUseProgram(0)
