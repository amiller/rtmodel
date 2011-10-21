import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.ARB.vertex_buffer_object import *

TEXTURE_TARGET = GL_TEXTURE_RECTANGLE


class PointModel(object):
    """A model of a 3D object represented as a 3D point cloud.
    """

    def __init__(self, xyz=None, normals=None, RT=np.eye(4,dtype='f')):
        """
        Args:
           xyz: numpy array (n by 3)
        """
        self.xyz = xyz.astype('f') if xyz is not None else xyz
        self.normals = normals.astype('f') if normals is not None else normals
        self.RT = RT.astype('f') if RT is not None else RT
        self._initialized = False


    def create_buffers(self):
        if self._initialized: return

        self.rgbtex = glGenTextures(1)
        glBindTexture(TEXTURE_TARGET, self.rgbtex)
        glTexImage2D(TEXTURE_TARGET,0,GL_RGB,640,480,0,GL_RGB,
                     GL_UNSIGNED_BYTE,None)

        self._depth = np.empty((480,640,3),np.int16)
        self._depth[:,:,1], self._depth[:,:,0] = np.mgrid[:480,:640]
        self.normalsbuf, self.xyzbuf, self.rgbabuf = glGenBuffersARB(3)

        glBindBufferARB(GL_ARRAY_BUFFER_ARB, self.xyzbuf)
        glBufferDataARB(GL_ARRAY_BUFFER_ARB, 640*480*3*4, None,GL_DYNAMIC_DRAW)

        glBindBufferARB(GL_ARRAY_BUFFER_ARB, self.rgbabuf)
        glBufferDataARB(GL_ARRAY_BUFFER_ARB, 640*480*4*4, None,GL_DYNAMIC_DRAW)

        glBindBufferARB(GL_ARRAY_BUFFER_ARB, self.normalsbuf)
        glBufferDataARB(GL_ARRAY_BUFFER_ARB, 640*480*3*4, None,GL_DYNAMIC_DRAW)

        self.update_buffers()
        self._initialized = True        


    def __del__(self):
        if self._initialized:
            glDeleteBuffersARB(3, [self.rgbabuf, self.normalsbuf, self.xyzbuf])
            glDeleteTextures((self.rgbtex,))


    def update_buffers(self):
        XYZ = self.xyz
        N = self.normals
        RGBA = None

        if XYZ is None: XYZ = np.zeros((0,3),'f')
        # TODO make this more elegant, coerce RGBA to match XYZ somehow
        assert XYZ.dtype == np.float32
        assert RGBA is None or RGBA.dtype == np.float32
        assert XYZ.shape[1] == 3
        assert RGBA is None or RGBA.shape[1] == 4
        assert RGBA is None or XYZ.shape[0] == RGBA.shape[0]
        self.XYZ = XYZ
        self.RGBA = RGBA
        self.N = N

        glBindBufferARB(GL_ARRAY_BUFFER_ARB, self.xyzbuf)
        glBufferSubDataARB(GL_ARRAY_BUFFER_ARB, 0, XYZ.shape[0]*3*4, XYZ)
        if not RGBA is None:
            glBindBufferARB(GL_ARRAY_BUFFER_ARB, self.rgbabuf)
            glBufferSubDataARB(GL_ARRAY_BUFFER_ARB, 0, XYZ.shape[0]*4*4, RGBA)
        elif not N is None:
            glBindBufferARB(GL_ARRAY_BUFFER_ARB, self.normalsbuf)
            glBufferSubDataARB(GL_ARRAY_BUFFER_ARB, 0, XYZ.shape[0]*3*4, N)

        glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0)


    def update_model(self):
        pass


    def draw(self):
        self.create_buffers()

        
        glPushMatrix()
        glMultMatrixf(self.RT.transpose())

        RGBA = None
        N = self.normals
        XYZ = self.xyz

        glBindBufferARB(GL_ARRAY_BUFFER_ARB, self.xyzbuf)
        glVertexPointerf(None)
        glEnableClientState(GL_VERTEX_ARRAY)
        if not RGBA is None:
            glBindBufferARB(GL_ARRAY_BUFFER_ARB, self.rgbabuf)
            glColorPointer(4, GL_FLOAT, 0, None)
            glEnableClientState(GL_COLOR_ARRAY)         
        if not N is None:
            glBindBufferARB(GL_ARRAY_BUFFER_ARB, self.normalsbuf)
            glNormalPointer(GL_FLOAT, 0, None)
            glEnableClientState(GL_NORMAL_ARRAY)
        if not XYZ is None:
            # Draw the points
            glDrawElementsui(GL_POINTS, np.arange(len(XYZ)))
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0)
        glDisable(GL_BLEND)

        glPopMatrix()
