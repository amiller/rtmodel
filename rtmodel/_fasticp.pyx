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
                 np.float32_t *normB,
                 int length,
                 np.float32_t *uv,
                 np.float32_t *matKK,
                 float dist2,
                 float normthresh,
                 np.float32_t A[6][6],
                 np.float32_t B[6]):

    cdef int i, _blank
    cdef float uw, vw, w, d, zz, normdot
    cdef int u, v
    cdef float xA, yA, zA
    cdef float nxA, nyA, nzA
    cdef float nxB, nyB, nzB
    cdef float xB, yB, zB
    cdef float c[3]
    cdef float sse = 0.
    cdef int npairs = 0


    for i in range(length):

        # Iterate over point sample in B
        xB = xyzB[i*3+0];
        yB = xyzB[i*3+1];
        zB = xyzB[i*3+2];

        # Iterate over point sample in B
        nxB = normB[i*3+0];
        nyB = normB[i*3+1];
        nzB = normB[i*3+2];

        # For each point in B, compute the projected coordinates in A
        uw = matKK[ 0]*xB + matKK[ 1]*yB + matKK[ 2]*zB + matKK[ 3];
        vw = matKK[ 4]*xB + matKK[ 5]*yB + matKK[ 6]*zB + matKK[ 7];
        w  = matKK[12]*xB + matKK[13]*yB + matKK[14]*zB + matKK[15];
	
        # Check that the normals in camera space are negative
        #if (zz > 0) ^ (w >= 0): continue
	#if w >= 0: continue
        w = 1./w
        u = <int> (uw * w)
        v = <int> (vw * w)

        # Discard if coordinates lie outside the image
        if not 0 <= u < width: continue
        if not 0 <= v < height: continue

        # Discard masked-out pixels
        if maskA[v*width + u] == 0: continue

        # Look up the point values in A
        xA = xyzA[(v*width + u)*3+0]
        yA = xyzA[(v*width + u)*3+1]
        zA = xyzA[(v*width + u)*3+2]

        # Look up the normal vectors in A
        nxA = normA[(v*width + u)*3+0];
        nyA = normA[(v*width + u)*3+1];
        nzA = normA[(v*width + u)*3+2];

        # Reject points if the normals don't match
        normdot = nxA * nxB + nyA * nyB + nzA * nzB
        if normdot < normthresh: continue

        # The following is taken directly from
        # pr2_fasticp.cc provided by Szymon Rusinkiewicz
        # http://www.cs.princeton.edu/~smr/software/pr2/
        d = (xB - xA) * nxA + \
            (yB - yA) * nyA + \
            (zB - zA) * nzA;

        # Reject outliers
        #print d*d
        if d*d > dist2: continue
        sse += d*d
        npairs += 1

        # Diagnostic: output u,v to check overlap
        #uv[i*6+0] = u
        #uv[i*6+1] = v
        uv[i*6+0] = xA
        uv[i*6+1] = yA
        uv[i*6+2] = zA
        uv[i*6+3] = xB
        uv[i*6+4] = yB
        uv[i*6+5] = zB

        c[0] = yA*nzA - zA*nyA
        c[1] = zA*nxA - xA*nzA
        c[2] = xA*nyA - yA*nxA

        # Accumulate 6x6 and 6x1 matrices
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

    return sse, npairs


def _fasticp(np.ndarray[np.float32_t, ndim=3, mode='c'] xyzA,
             np.ndarray[np.float32_t, ndim=3, mode='c'] normA,
             np.ndarray[np.uint8_t, ndim=2, mode='c'] maskA,
             np.ndarray[np.float32_t, ndim=2, mode='c'] xyzB,
             np.ndarray[np.float32_t, ndim=2, mode='c'] normB,
             np.ndarray[np.float32_t, ndim=2, mode='c'] matKK,
             float dist2, float normthresh):
    height, width = maskA.shape[0], maskA.shape[1]
    length = xyzB.shape[0]
    assert xyzA.shape[0] == normA.shape[0] == height, 'xyzA normA shape height'
    assert xyzA.shape[1] == normA.shape[1] == width, 'xyzA normA shape'
    assert xyzA.shape[2] == normA.shape[2] == xyzB.shape[1] == 3, 'xyz norm xyzB shape'
    assert matKK.shape[0] == matKK.shape[1] == 4, 'matKK shape'

    # Build an intermediate value UV coordinates matrix (for debugging)
    cdef np.ndarray[np.float32_t, ndim=2, mode='c'] uv = \
         np.zeros((length, 6), dtype='f')

    # Build the accumulator matrices A (6x6) and B (6x1)
    cdef np.ndarray[np.float32_t, ndim=2, mode='c'] A = \
         np.zeros((6, 6), dtype='f')
    cdef np.ndarray[np.float32_t, ndim=1, mode='c'] B = \
         np.zeros((6,), dtype='f')

    sse, npairs = _c_fast_icp(<np.float32_t *> xyzA.data,
                              <np.float32_t *> normA.data,
                              <np.uint8_t *> maskA.data,
                              height, width,
                              <np.float32_t *> xyzB.data,
                              <np.float32_t *> normB.data,
                              length,
                              <np.float32_t *> uv.data,
                              <np.float32_t *> matKK.data,
                              dist2, 
                               normthresh,
                              <np.float32_t (*)[6]> A.data,
                              <np.float32_t *> B.data
                              )

    return sse, npairs, uv, A, B


