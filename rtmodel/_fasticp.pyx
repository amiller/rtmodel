cimport numpy as np
import numpy as np
import cython
from pointmodel import PointModel
import transformations


cdef extern from "stdlib.h":
    double drand48()
    void srand48(long int seedval)

cdef extern from "time.h":
    long int time(int)

srand48(time(0))


@cython.cdivision(True)
cdef _c_fast_icp(np.float32_t *xyzA,
                 np.float32_t *normA,
                 np.uint8_t *maskA,
                 int height, int width,
                 np.float32_t *xyzB,
                 int length,
                 np.float32_t *uv,
                 np.float32_t *matB,
                 np.float32_t *matKKBtoA,
                 float rate, float dist2,
                 np.float32_t A[6][6],
                 np.float32_t B[6]):

    cdef int i, _blank
    cdef float x, y, z, uw, vw, w, d, zz
    cdef int u, v
    cdef float xA, yA, zA
    cdef float nxA, nyA, nzA
    cdef float xB, yB, zB
    cdef float c[3]
    cdef float err = 0.
    cdef int npairs = 0

    # Sample from the set of points A
    if rate <= 0.: rate = 0.001
    cdef float two_over_rate = 2.0 / rate

    i = 0
    while True:
        i += int(1 + two_over_rate * drand48())
        if not i < length:
            break

        # Get the coordinates in the range image A
        x = xyzB[i*3+0];
        y = xyzB[i*3+1];
        z = xyzB[i*3+2];
        uw = matKKBtoA[ 0]*x + matKKBtoA[ 1]*y + matKKBtoA[ 2]*z + matKKBtoA[ 3];
        vw = matKKBtoA[ 4]*x + matKKBtoA[ 5]*y + matKKBtoA[ 6]*z + matKKBtoA[ 7];
        w  = matKKBtoA[12]*x + matKKBtoA[13]*y + matKKBtoA[14]*z + matKKBtoA[15];
	
        # Check that the normals in camera space are negative
        #if (zz > 0) ^ (w >= 0): continue
	#if w >= 0: continue
        w = 1./w
        u = <int> (uw * w)
        v = <int> (vw * w)

        # Discard if outside the image
        if not 0 <= u < width: continue
        if not 0 <= v < height: continue

        if maskA[v*width + u] == 0: continue

        xB = matB[ 0]*x + matB[ 1]*y + matB[ 2]*z + matB[ 3];
        yB = matB[ 4]*x + matB[ 5]*y + matB[ 6]*z + matB[ 7];
        zB = matB[ 8]*x + matB[ 9]*y + matB[10]*z + matB[11];

        xA = xyzA[(v*width + u)*3+0]
        yA = xyzA[(v*width + u)*3+1]
        zA = xyzA[(v*width + u)*3+2]

        nxA = normA[(v*width + u)*3+0];
        nyA = normA[(v*width + u)*3+1];
        nzA = normA[(v*width + u)*3+2];

        # The following is borrowed directly from
        # pr2_fasticp.cc provided by Szymon Rusinkiewicz
        # http://www.cs.princeton.edu/~smr/software/pr2/
        d = (xB - xA) * nxA + \
            (yB - yA) * nyA + \
            (zB - zA) * nzA;

        # Diagnostic: output u,v to check overlap

        if d*d > dist2: continue
        err += d*d
        npairs += 1

        #uv[_blank*6+0] = xA
        #uv[_blank*6+1] = yA
        #uv[_blank*6+2] = zA
        #uv[_blank*6+3] = xB
        #uv[_blank*6+4] = yB
        #uv[_blank*6+5] = zB

        c[0] = yA*nzA - zA*nyA
        c[1] = zA*nxA - xA*nzA
        c[2] = xA*nyA - yA*nxA

        B[0] += d * c[0];
        B[1] += d * c[1];
        B[2] += d * c[2];
        B[3] += d * nxA
        B[4] += d * nyA
        B[5] += d * nzA
        A[0][0] += c[0] * c[0];
        A[0][1] += c[0] * c[1];
        A[0][2] += c[0] * c[2];
        A[0][3] += c[0] * nxA
        A[0][4] += c[0] * nyA
        A[0][5] += c[0] * nzA
        A[1][1] += c[1] * c[1];
        A[1][2] += c[1] * c[2];
        A[1][3] += c[1] * nxA
        A[1][4] += c[1] * nyA
        A[1][5] += c[1] * nzA
        A[2][2] += c[2] * c[2];
        A[2][3] += c[2] * nxA
        A[2][4] += c[2] * nyA
        A[2][5] += c[2] * nzA
        A[3][3] += nxA * nxA
        A[3][4] += nxA * nyA
        A[3][5] += nxA * nzA
        A[4][4] += nyA * nyA
        A[4][5] += nyA * nzA
        A[5][5] += nzA * nzA

    err /= npairs
    return err, npairs


def _fasticp(np.ndarray[np.float32_t, ndim=3, mode='c'] xyzA,
             np.ndarray[np.float32_t, ndim=3, mode='c'] normA,
             np.ndarray[np.uint8_t, ndim=2, mode='c'] maskA,
             np.ndarray[np.float32_t, ndim=2, mode='c'] xyzB,
             np.ndarray[np.float32_t, ndim=2, mode='c'] matB,
             np.ndarray[np.float32_t, ndim=2, mode='c'] matKKBtoA,
             float rate, float dist2):
    height, width = maskA.shape[0], maskA.shape[1]
    length = xyzB.shape[0]
    assert xyzA.shape[0] == normA.shape[0] == height, 'xyzA normA shape height'
    assert xyzA.shape[1] == normA.shape[1] == width, 'xyzA normA shape'
    assert xyzA.shape[2] == normA.shape[2] == xyzB.shape[1] == 3, 'xyz norm xyzB shape'
    assert matKKBtoA.shape[0] == matKKBtoA.shape[1] == 4, 'matKKBtoA shape'
    assert matB.shape[0] == matB.shape[1] == 4, 'matB shape'


    # Build an intermediate value UV coordinates matrix
    cdef np.ndarray[np.float32_t, ndim=2, mode='c'] uv = \
         np.zeros((length, 6), dtype='f')

    # Build the accumulator matrices A (6x6) and B (6x1)
    cdef np.ndarray[np.float32_t, ndim=2, mode='c'] A = \
         np.zeros((6, 6), dtype='f')
    cdef np.ndarray[np.float32_t, ndim=1, mode='c'] B = \
         np.zeros((6,), dtype='f')

    err, npairs = _c_fast_icp(<np.float32_t *> xyzA.data,
                              <np.float32_t *> normA.data,
                              <np.uint8_t *> maskA.data,
                              height, width,
                              <np.float32_t *> xyzB.data,
                              length,
                              <np.float32_t *> uv.data,
                              <np.float32_t *> matB.data,
                              <np.float32_t *> matKKBtoA.data,
                              rate, dist2,
                              <np.float32_t (*)[6]> A.data,
                              <np.float32_t *> B.data
                              )

    return err, npairs, uv, A, B


