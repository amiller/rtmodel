import numpy as np
from OpenGL.GL import *
import calibkinect
import _volume


def depth_inds(mat, X, Y, Z):
    x = X*mat[0,0] + Y*mat[0,1] + Z*mat[0,2] + mat[0,3]
    y = X*mat[1,0] + Y*mat[1,1] + Z*mat[1,2] + mat[1,3]
    z = X*mat[2,0] + Y*mat[2,1] + Z*mat[2,2] + mat[2,3]
    w = X*mat[3,0] + Y*mat[3,1] + Z*mat[3,2] + mat[3,3]
    w = 1/w
    return x*w, y*w, z*w


class Volume(object):
    def __init__(self, RT=np.eye(4, dtype='f')):

        # Number of voxel cells per side
        self.N = N = 512

        # Volume size, in meters
        width = 2.0

        self.RT = np.eye(4)
        self.KK = np.array([[N/width, 0, 0, N/2],
                            [0, N/width, 0, N/2],
                            [0, 0, N/width, N/2],
                            [0, 0, 0, 1]], 'f')

        # Saturated signed distance metric
        self.SD = np.zeros([N,N,N], 'i2')

        # Weights
        self.W = np.zeros([N,N,N], 'u2')

        # Constants
        self.MAX_D = 0.02


    def render_bounds(self):
        # Draw the eight corners
        try:
            glMatrixMode(GL_MODELVIEW)
            glPushMatrix()
            glPushAttrib(GL_ALL_ATTRIB_BITS)
            glMultMatrixf(self.RT.transpose())
            glMultMatrixf(np.linalg.inv(self.KK).transpose())
            glScale(self.N, self.N, self.N)
            glLineWidth(1)
            glColor(1,1,1)
            glBegin(GL_LINES)
            glVertex(0,0,0); glVertex(0,0,1);
            glVertex(0,0,0); glVertex(0,1,0);
            glVertex(0,0,0); glVertex(1,0,0);
            glVertex(1,1,0); glVertex(0,1,0);
            glVertex(1,1,0); glVertex(1,0,0);
            glVertex(1,1,0); glVertex(1,1,1);
            glVertex(0,1,1); glVertex(1,1,1);
            glVertex(0,1,1); glVertex(0,0,1);
            glVertex(0,1,1); glVertex(0,1,0);
            glVertex(1,0,1); glVertex(0,0,1);
            glVertex(1,0,1); glVertex(0,0,1);
            glVertex(1,0,1); glVertex(1,1,1);
            glVertex(1,0,1); glVertex(1,0,0);
            glEnd()
        finally:
            glPopAttrib(GL_ALL_ATTRIB_BITS)            
            glPopMatrix()


    def distance_transform(self, range_image):
        # Find the reference depth for each voxel, and the sampled depth
        cam = range_image.camera
        mat = np.eye(4, dtype='f')
        mat = np.dot(mat, cam.KK)
        mat = np.dot(mat, np.linalg.inv(cam.RT))
        mat = np.dot(mat, self.RT)
        mat = np.dot(mat, np.linalg.inv(self.KK))
        mat = mat.astype('f')

        depth = 0.001 * range_image.depth
        _volume._volume(depth, self.SD, self.W, mat, self.MAX_D)


    def distance_transform_numpy(self, range_image):
        MAX_D = self.MAX_D
        
        global x, y, d

        # Consider the center points of each candidate voxel
        X,Y,Z = np.mgrid[:self.N,:self.N,:self.N] + 0.5

        # Find the reference depth for each voxel, and the sampled depth
        cam = range_image.camera
        mat = np.eye(4, dtype='f')
        mat = np.dot(mat, cam.KK)
        mat = np.dot(mat, np.linalg.inv(cam.RT))
        mat = np.dot(mat, self.RT)
        mat = np.dot(mat, np.linalg.inv(self.KK))

        x,y,dref = depth_inds(mat, X,Y,Z)
        dref = 1./dref

        depth_ = 0.001 * range_image.depth

        import scipy.ndimage
        d = scipy.ndimage.map_coordinates(depth_, (y,x), order=0,
                                          prefilter=False,
                                          cval=-np.inf)

        # dref is the expected range (meters) to the center of the voxel
        # d is the observed range value

        # TODO add the saturated distance ramp
        # TODO add the weight function
        sd = d-dref
        self.d = np.maximum(0, np.minimum(10, d))
        self.dref = dref
        self.W[:,:,:] = (-MAX_D < sd) & (sd < MAX_D)
        self.SD[:,:,:] = np.maximum(-MAX_D, np.minimum(MAX_D, sd))
