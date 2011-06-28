import numpy as np
import calibkinect
import pointmodel


class RangeImage(object):
    """Depth image, numpy 2D array ('f').
    - Depth values of zero indicate no good measurement
    - Normal computation is performed on the original
    """
    def __init__(self, depth, camera=None):
        self.depth = depth
        self.camera = camera

    def compute_normals(self, rect=((0,0),(640,480)), win=7):
        """
        """
        assert self.depth.dtype == np.float32
        from scipy.ndimage.filters import uniform_filter
        (l,t),(r,b) = rect
        v,u = np.mgrid[t:b,l:r]
        depth = self.depth
        depth = depth[v,u]
        depth[depth==0] = -1e8  # 2047
        depth = calibkinect.recip_depth_openni(depth.astype('u2'))
        self.drecip = depth
        depth = uniform_filter(depth, win)
        self.duniform = depth
        
        dx = (np.roll(depth,-1,1) - np.roll(depth,1,1))/2
        dy = (np.roll(depth,-1,0) - np.roll(depth,1,0))/2

        X,Y,Z,W = -dx, -dy, 0*dy+1, -(-dx*u + -dy*v + depth).astype(np.float32)

        if self.camera is not None:
            mat = self.camera.KK.transpose()
        else:
            mat = np.eye(4,'f')

        x = X*mat[0,0] + Y*mat[0,1] + Z*mat[0,2] + W*mat[0,3]
        y = X*mat[1,0] + Y*mat[1,1] + Z*mat[1,2] + W*mat[1,3]
        z = X*mat[2,0] + Y*mat[2,1] + Z*mat[2,2] + W*mat[2,3]

        w = np.sqrt(x*x + y*y + z*z)
        w[z<0] *= -1
        weights = z*0+1
        weights[depth<-1000] = 0
        weights[(z/w)<.1] = 0

        self.normals = np.dstack((x/w,y/w,z/w))
        self.weights = weights


    def pointmodel(self):
        if self.camera is not None:
            mat = np.linalg.inv(self.camera.KK)
        else:
            mat = np.eye(4,'f')

        v,u = np.mgrid[:480,:640].astype('f')
        depth = self.depth
        depth = calibkinect.recip_depth_openni(depth.astype('u2'))
        X,Y,Z,W = u,v,depth,1

        x = X*mat[0,0] + Y*mat[0,1] + Z*mat[0,2] + W*mat[0,3]
        y = X*mat[1,0] + Y*mat[1,1] + Z*mat[1,2] + W*mat[1,3]
        z = X*mat[2,0] + Y*mat[2,1] + Z*mat[2,2] + W*mat[2,3]
        w = X*mat[3,0] + Y*mat[3,1] + Z*mat[3,2] + W*mat[3,3]

        return pointmodel.PointModel(np.dstack((x/w,y/w,z/w)).reshape(-1,3))

