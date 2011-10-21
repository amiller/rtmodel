cimport numpy as np
import numpy as np
import cython

@cython.cdivision(True)
cdef _volume_c(np.float32_t *depth,
               int height, int width,
               np.int16_t *SD,
               np.uint16_t *W,
               np.float32_t m[16],    
               int nx, int ny, int nz,
               float MAX_D):

    cdef int iX, iY, iZ
    cdef float X, Y, Z
    cdef float x, y, z, w

    cdef float d, dref
    cdef float sd, sw
    cdef int i = 0
    cdef int ix, iy

    for iX in range(nx):
        X = (iX+<float>0.5)
        for iY in range(ny):
            Y = (iY+<float>0.5)
            for iZ in range(nz):
                Z = (iZ+<float>0.5)

                x = X*m[ 0] + Y*m[ 1] + Z*m[ 2] + m[ 3]
                y = X*m[ 4] + Y*m[ 5] + Z*m[ 6] + m[ 7]
                z = X*m[ 8] + Y*m[ 9] + Z*m[10] + m[11]
                w = X*m[12] + Y*m[13] + Z*m[14] + m[15]

                w = 1/w if not w == 0 else 0
                x = x * w
                y = y * w
                z = z * w

                dref = 1./z

                ix = <int> x
                iy = <int> y

                ix = 0 if ix < 0 else ix
                iy = 0 if iy < 0 else iy
                ix = width-1 if ix >= width else ix
                iy = height-1 if iy >= height else iy

                d = depth[iy*width+ix]

                sd = d-dref

                # Clamp and apply weighting function
                sw = 1
                if sd < -MAX_D:
                    sw = 0
                    sd = -MAX_D
                elif sd > MAX_D:
                    sw = 0
                    sd = MAX_D

                sd = sd * 32767. / MAX_D

                SD[i] = <np.int16_t> sd
                W[i] = <np.uint16_t> sw * 32767
                i += 1


def _volume(np.ndarray[np.float32_t, ndim=2, mode='c'] depth,
            np.ndarray[np.int16_t, ndim=3, mode='c'] SD,
            np.ndarray[np.uint16_t, ndim=3, mode='c'] W,
            np.ndarray[np.float32_t, ndim=2, mode='c'] M,
            MAX_D):
    assert depth.shape[0] == 480 and depth.shape[1] == 640
    assert (SD.shape[0] == W.shape[0] and SD.shape[1] == W.shape[1]
            and SD.shape[2] == W.shape[2])
    assert M.shape[0] == M.shape[1] == 4
    assert M.dtype == depth.dtype == np.float32
    assert SD.dtype == np.int16
    assert W.dtype == np.uint16

    _volume_c(<np.float32_t *> depth.data,
              480, 640,
              <np.int16_t *> SD.data,
              <np.uint16_t *> W.data,
              <np.float32_t *> M.data,
              SD.shape[0], SD.shape[1], SD.shape[2],
              MAX_D)
