import numpy as np
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.ARB.vertex_buffer_object import *

TEXTURE_TARGET = GL_TEXTURE_RECTANGLE


class PointModel(object):
    """A model of a 3D object represented as a 3D point cloud.
    """

    def __init__(self, xyz=None, norm=None, RT=np.eye(4,dtype='f')):
        """
        Args:
           xyz: numpy array (n by 3)
        """
        self.xyz = xyz
        self.norm = norm
        self.RT = RT
        self._initialized = False


    def create_buffers(self):
        if self._initialized: return

        self.rgbtex = glGenTextures(1)
        glBindTexture(TEXTURE_TARGET, self.rgbtex)
        glTexImage2D(TEXTURE_TARGET,0,GL_RGB,640,480,0,GL_RGB,
                     GL_UNSIGNED_BYTE,None)

        self._depth = np.empty((480,640,3),np.int16)
        self._depth[:,:,1], self._depth[:,:,0] = np.mgrid[:480,:640]
        self.xyzbuf = glGenBuffersARB(1)
        glBindBufferARB(GL_ARRAY_BUFFER_ARB, self.xyzbuf)
        glBufferDataARB(GL_ARRAY_BUFFER_ARB, 640*480*3*4, None,GL_DYNAMIC_DRAW)
        self.rgbabuf = glGenBuffersARB(1)
        glBindBufferARB(GL_ARRAY_BUFFER_ARB, self.rgbabuf)
        glBufferDataARB(GL_ARRAY_BUFFER_ARB, 640*480*4*4, None,GL_DYNAMIC_DRAW)
        self.normalsbuf = glGenBuffersARB(1)
        glBindBufferARB(GL_ARRAY_BUFFER_ARB, self.normalsbuf)
        glBufferDataARB(GL_ARRAY_BUFFER_ARB, 640*480*4*3, None,GL_DYNAMIC_DRAW)

        self._initialized = True
        self.update_buffers()


    def __close__(self):
        if self._initialized:
            glDeleteBuffersARB([self.rgbabuf, self.normalsbuf, self.xyzbuf])


    def update_buffers(self):
        XYZ = self.xyz
        N = self.norm
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
            glBufferSubDataARB(GL_ARRAY_BUFFER_ARB, 0, XYZ.shape[0]*4*3, N)            
        glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0)


    def update_model(self):
        pass


    def draw(self):        
        self.create_buffers()

        glPushMatrix()
        glMultMatrixf(self.RT.transpose())

        RGBA = None
        N = self.norm
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
            glPointSize(2)
            glDrawElementsui(GL_POINTS, np.arange(len(XYZ)))
        glDisableClientState(GL_COLOR_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)            
        glDisableClientState(GL_VERTEX_ARRAY)
        glBindBufferARB(GL_ARRAY_BUFFER_ARB, 0)
        glDisable(GL_BLEND)

        glPopMatrix()
