cimport numpy as np
import numpy as np
import rangeimage
import pointmodel
cimport cython

cdef np.float32_t *make_lut():
    global _lut
    _lut = np.arange(5000).astype('f')
    cdef np.ndarray[np.float32_t, ndim=1, mode='c'] lut = _lut
    lut[1:] = 1000./lut[1:]
    lut[0] = -1e8
    return <np.float32_t *> lut.data

cdef np.float32_t *lut = make_lut()

@cython.cdivision(True)
cdef _point_model(np.uint16_t *depth,
                  np.float32_t *out,
                  np.float32_t *xyz,
                  np.float32_t *mat,
                  int width, int height):
    cdef int i = 0
    cdef int n = 0
    cdef int w, h
    cdef int x, y
    cdef int D
    cdef float X, Y, Z, W, z
    
    for y in range(height):
        for x in range(width):
            D = depth[i]
            if D > 0:
                z = lut[D]
                X = mat[ 0]*x + mat[ 1]*y + mat[ 2]*z + mat[ 3];
                Y = mat[ 4]*x + mat[ 5]*y + mat[ 6]*z + mat[ 7];
                Z = mat[ 8]*x + mat[ 9]*y + mat[10]*z + mat[11];
                W = mat[12]*x + mat[13]*y + mat[14]*z + mat[15];
                W = 1./W
                X *= W
                Y *= W
                Z *= W
                out[3*n+0] = X
                out[3*n+1] = Y
                out[3*n+2] = Z
                xyz[3*i+0] = X
                xyz[3*i+1] = Y
                xyz[3*i+2] = Z
                n += 1
            i += 1
    return n


class RangeImage(rangeimage.RangeImage):
    def point_model(self, do_compute_normals=False):        
        cdef np.ndarray[np.uint16_t, ndim=2, mode='c'] depth = self.depth.astype('u2')
        width, height = depth.shape[1], depth.shape[0]

        cdef np.ndarray[np.float32_t, ndim=2, mode='c'] out = np.empty((width*height, 3), 'f')
        cdef np.ndarray[np.float32_t, ndim=3, mode='c'] xyz = np.zeros((height, width, 3), 'f')

        cdef np.ndarray[np.float32_t, ndim=2, mode='c'] mat
        if self.camera is not None:
            mat = np.ascontiguousarray(np.linalg.inv(self.camera.KK))
        else:
            mat = np.eye(4,'f')

        num = _point_model(<np.uint16_t *> depth.data,
                           <np.float32_t *> out.data,
                           <np.float32_t *> xyz.data,
                           <np.float32_t *> mat.data,
                           width, height)
        self.xyz = xyz

        if do_compute_normals:
            self.compute_normals()
            mask = depth > 0
            n = np.ascontiguousarray(self.normals[mask,:])
        else:
            n = None

        return pointmodel.PointModel(out[:num,:], n, self.camera.RT)

    def point_model_py(self, *args, **kwargs):
        return super(RangeImage, self).point_model(*args, **kwargs)

# Monkey patch ourselves right in there
rangeimage.RangeImage = RangeImage
