import numpy as np
import calibkinect
from OpenGL.GL import *
from wxpy3d.opengl_state import opengl_state

class Camera(object):

    def __init__(self, KK, RT=np.eye(4, dtype='f')):
        """
        Args:
            KK: 4x4 numpy matrix ('f')
                intrinsic parameters, origin remains at 0
            RT: 4x4 numpy matrix ('f')
                rotation and translation
        """
        self.KK = np.ascontiguousarray(KK.astype('f'))
        self.RT = np.ascontiguousarray(RT.astype('f'))

    def render_frustum(self):
        with opengl_state():
            glMultMatrixf(self.RT.transpose())
            glScale(0.2, 0.2, 0.2)

            # Draw a little box, that's all
            glBegin(GL_LINES)
            glColor(1,0,0); glVertex(0,0,0); glVertex(1,0,0)
            glColor(0,1,0); glVertex(0,0,0); glVertex(0,1,0)
            glColor(0,0,1); glVertex(0,0,0); glVertex(0,0,1)
            glColor(1,1,1); glVertex(0,0,0); glVertex(0,0,-1)
            glEnd()


def kinect_camera():
    return Camera(calibkinect.projection())
