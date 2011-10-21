import numpy as np
import calibkinect
import pointmodel


class RangeImage(object):
    """Depth image, numpy 2D array ('f').
    - Depth units are in millimeters, just like OpenNI
    - Values of zero indicate no good measurement
    - Normals computation is performed on the
        1.0/meters image (recip_depth_openni)

    Invariants:
       The internal data does not depend on self.camera.RT so self.camera.RT
       can change.

       The @camera.KK intrinsic matrix must not change, because internal
       state @xyz and @normals DO depend on it. If the RT matrix has
       to change, you could just rebuild them with @compute_normals()


    NOTE: Camera.KK is backwards compared to the convention of RT matrices:
          [u,v,1]' = KK [x, y, z, 1]
      Where [u,v,1] is the storage-specific image coordinates, [x,y,z,1] are
      absolute world coordinates. The convention otherwise is:
          [x,y,z,1]' = RT [xp,yp,zp,1] where [xp,yp,zp,1] are the values stored
      in data and applying RT to them produces world coordinates

        

    NOTE: The fields self.normals and self.xyz are images
    and they still need to apply RT.
    """
    def __init__(self, depth, camera=None):
        self.depth = depth
        self.camera = camera


    def compute_normals(self, rect=((0,0),(640,480)), win=7):
        """Computes the normals
        """
        assert self.depth.dtype == np.float32
        from scipy.ndimage.filters import uniform_filter
        (l,t),(r,b) = rect
        v,u = np.mgrid[t:b,l:r]
        depth = self.depth
        depth = depth[v,u]
        #depth[depth==0] = -1e8  # 2047
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

        self.normals = np.ascontiguousarray(np.dstack((x/w,y/w,z/w)))
        self.weights = weights


    def point_model(self, do_compute_normals=False):
        """
        Effects:
            self.xyz will contain a 2D grid of XYZ points in camera
            relative euclidean coordinates. It's still necessary to
            apply self.camera.RT.
        """
        if self.camera is not None:
            mat = np.linalg.inv(self.camera.KK)
        else:
            mat = np.eye(4,'f')

        v,u = np.mgrid[:480,:640].astype('f')
        depth = self.depth
        depth = calibkinect.recip_depth_openni(depth.astype('u2'))

        if 'weights' in self.__dict__:
            mask = (depth > 0) & (self.weights > 0)
        else:
            mask = depth > 0
        

        X,Y,Z,W = u,v,depth,1

        x = X*mat[0,0] + Y*mat[0,1] + Z*mat[0,2] + W*mat[0,3]
        y = X*mat[1,0] + Y*mat[1,1] + Z*mat[1,2] + W*mat[1,3]
        z = X*mat[2,0] + Y*mat[2,1] + Z*mat[2,2] + W*mat[2,3]
        w = X*mat[3,0] + Y*mat[3,1] + Z*mat[3,2] + W*mat[3,3]

        if do_compute_normals:
            self.compute_normals()
            n = np.ascontiguousarray(self.normals[mask,:])
        else:
            n = None

        self.xyz = np.ascontiguousarray(np.dstack((x/w,y/w,z/w)))
        return pointmodel.PointModel(np.ascontiguousarray(self.xyz[mask,:]), n, self.camera.RT)
