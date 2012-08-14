import numpy as np
import calibkinect
import pointmodel
import scipy.ndimage
from camera import Camera


class RangeImage(object):
    """Depth image, numpy 2D array ('f').
    - Depth units are in millimeters, just like OpenNI
    - Values of zero indicate no good measurement
    - Normals computation is performed on the
        1.0/meters image (recip_depth_openni)
    - A Region-of-interest is represented using a 'rect' and 'mask'

    In addition to the depth data itself, this object may contain several
    intermediate stages of processing:
        - Filtered and background-subtracted depth
        - Normal vectors (world coordinates)
        - Metric XYZ (world coordinates)


    NOTE: Camera.KK is backwards compared to the convention of RT matrices:
          [u,v,1]' = KK [x, y, z, 1]
      Where [u,v,1] is the storage-specific image coordinates, [x,y,z,1] are
      absolute world coordinates. The convention otherwise is:
          [x,y,z,1]' = RT [xp,yp,zp,1] where [xp,yp,zp,1] are the values stored
      in data and applying RT to them produces world coordinates

    """
    def __init__(self, depth, camera, rect=None, mask=None):
        assert depth.dtype == np.uint16
        assert len(depth.shape) == 2
        self.depth = depth
        
        assert type(camera) is Camera
        self.camera = camera

        if rect is None:
            h,w = depth.shape
            rect = ((0,0),(w,h))
        else:
            (l,t),(r,b) = rect
        self.rect = rect

        if mask is None:
            mask = np.ones_like(depth).astype('u1').astype('bool')
        self.mask = mask
        self.weights = mask.astype('f')

        # The computed world points @xyz and @normals are associated with
        # a second transformation matrix, RT, which is used to allow the
        # model coordinates to change without recomputing the data.
        self.RT = np.eye(4,dtype='f')
        self.xyz = None
        self.normals = None

    def _inrange(self, lo, hi):
        return (self.depth>lo) & (self.depth<hi)  # background

    def threshold_and_mask(self, bg):
        """Modifies the values for mask, weight, and rect.
        """
        mask = self._inrange(bg['bgLo'], bg['bgHi'])
        dec = 3
        dil = scipy.ndimage.binary_erosion(mask[::dec,::dec],iterations=2)
        slices = scipy.ndimage.find_objects(dil)
        a,b = slices[0]
        (l,t),(r,b) = (b.start*dec-10,a.start*dec-10),(b.stop*dec+7,a.stop*dec+7)
        b += -(b-t)%16 # Make the rect into blocks of 16x16 (for the gpu)
        r += -(r-l)%16 #
        if t<0: t+= 16
        if l<0: l+= 16
        if r>=640: r-= 16
        if b>=480: b-= 16

        self.mask = mask
        self.weights = (self.depth > 0) & mask
        self.rect = ((l,t),(r,b))

    def filter(self, win=7):
        """Creates @depth_filtered and @depth_recip"""
        depth = from_rect(self.depth, self.rect)
        depth = np.ascontiguousarray(depth)
        depth = calibkinect.recip_depth_openni(depth)
        self.depth_recip = depth
        depth = scipy.ndimage.uniform_filter(depth,win)
        self.depth_filtered = depth

    def compute_normals(self):
        """Computes the normals, with KK *and* RT applied. The stored points
        are in global coordinates.
        """
        v,u = from_rect(np.mgrid, self.rect)
        if 'depth_filtered' in self.__dict__:
            depth = self.depth_filtered
        else:
            depth = calibkinect.recip_depth_openni(from_rect(self.depth, self.rect))

        dx = (np.roll(depth,-1,1) - np.roll(depth,1,1))/2
        dy = (np.roll(depth,-1,0) - np.roll(depth,1,0))/2

        X,Y,Z,W = -dx, -dy, 0*dy+1, -(-dx*u + -dy*v + depth).astype(np.float32) 

        mat = np.linalg.inv(self.camera.KK).transpose()

        x = X*mat[0,0] + Y*mat[0,1] + Z*mat[0,2] + W*mat[0,3]
        y = X*mat[1,0] + Y*mat[1,1] + Z*mat[1,2] + W*mat[1,3]
        z = X*mat[2,0] + Y*mat[2,1] + Z*mat[2,2] + W*mat[2,3]

        w = np.sqrt(x*x + y*y + z*z)
        w[z<0] *= -1
        x,y,z = (_ / w for _ in (x,y,z))

        weights = z*0+1
        weights[depth<-1000] = 0
        weights[~from_rect(self.mask,self.rect)] = 0
        weights[z<=.1] = 0
        weights[np.abs(dx)+np.abs(dy) > 10] = 0

        mat = self.camera.RT
        x_ = x*mat[0,0] + y*mat[0,1] + z*mat[0,2]
        y_ = x*mat[1,0] + y*mat[1,1] + z*mat[1,2]
        z_ = x*mat[2,0] + y*mat[2,1] + z*mat[2,2]

        #self.normals = np.empty(self.depth.shape+(3,), 'f')
        #self.normals[t:b,l:r] = np.dstack((x_,y_,z_))
        self.normals = np.ascontiguousarray(np.dstack((x_,y_,z_)))
        self.weights = weights


    def compute_points(self):
        mat = np.dot(self.camera.RT, self.camera.KK)

        # TODO: Make this faster by following the region of interest

        # Convert the depth to units of (1./meters)
        depth = calibkinect.recip_depth_openni(self.depth)
        v,u = np.mgrid[:480,:640].astype('f')
        X,Y,Z = u, v, depth

        x = X*mat[0,0] + Y*mat[0,1] + Z*mat[0,2] + mat[0,3]
        y = X*mat[1,0] + Y*mat[1,1] + Z*mat[1,2] + mat[1,3]
        z = X*mat[2,0] + Y*mat[2,1] + Z*mat[2,2] + mat[2,3]
        w = X*mat[3,0] + Y*mat[3,1] + Z*mat[3,2] + mat[3,3]
        w = 1/w
        self.xyz = np.ascontiguousarray(np.dstack((x*w,y*w,z*w)))
        

    def point_model(self):
        """
        Effects:
            self.xyz will contain a 2D grid of XYZ points in camera
            relative euclidean coordinates. It's still necessary to
            apply self.camera.RT.
        """
        assert self.xyz is not None, 'point_model() called but no metric points found \
                                      (try calling compute_points() first)'
        (l,t),(r,b) = self.rect

        # TODO: Settle on whether I want chips or embedded images
        mask = np.zeros_like(self.depth).astype('bool')
        mask[t:b,l:r] = self.weights > 0
        #mask = self.weights > 0
        xyz = self.xyz[mask,:]
        normals = self.normals[self.weights>0,:] if self.normals is not None else None
        return pointmodel.PointModel(xyz, normals, self.RT)


def from_rect(depth,rect):
    (l,t),(r,b) = rect
    return depth[t:b,l:r]
