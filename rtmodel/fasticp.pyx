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


@cython.cdivision(True)
cdef project_coords(np.float32_t *xyz, int length,
                    np.float32_t *mat,
                    np.float32_t *uv):
    cdef int i, j
    cdef float x, y, z, uw, vw, w, u, v
    for i in range(0, length):
        x = xyz[i*3+0];
        y = xyz[i*3+1];
        z = xyz[i*3+2];
        uw = mat[ 0]*x + mat[ 1]*y + mat[ 2]*z + mat[ 3];
        vw = mat[ 4]*x + mat[ 5]*y + mat[ 6]*z + mat[ 7];
        w  = mat[12]*x + mat[13]*y + mat[14]*z + mat[15];
        u = uw * (1./w);
        v = vw * (1./w);
        uv[i*2+0] = u;
        uv[i*2+1] = v;


@cython.cdivision(True)
cdef _c_fast_icp(np.float32_t *xyzA,
                 np.float32_t *normA,
                 np.float32_t *matA,
                 int length,
                 np.float32_t *xyzB,
                 np.float32_t *matKKB,
                 np.uint8_t *mask,
                 np.float32_t *uv,
                 int height, int width,
                 int samples, float dist2,
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
    for _blank in range(samples):
        i = <int> (drand48() * length);

        # Get the coordinates in the range image B
        x = xyzA[i*3+0];
        y = xyzA[i*3+1];
        z = xyzA[i*3+2];
        uw = matKKB[ 0]*x + matKKB[ 1]*y + matKKB[ 2]*z + matKKB[ 3];
        vw = matKKB[ 4]*x + matKKB[ 5]*y + matKKB[ 6]*z + matKKB[ 7];
        zz = matKKB[ 8]*x + matKKB[ 9]*y + matKKB[10]*z + matKKB[11];
        w  = matKKB[12]*x + matKKB[13]*y + matKKB[14]*z + matKKB[15];

        # Check that the normals in camera space are negative
        if (zz < 0) ^ (w < 0): continue
        w = 1./w
        u = <int> (uw * w)
        v = <int> (vw * w)

        # Discard if outside the image
        if not 0 <= u < width: continue
        if not 0 <= v < height: continue

        if mask[v*width + u] == 0: continue

        xB = xyzB[(v*width + u)*3 + 0]
        yB = xyzB[(v*width + u)*3 + 1]
        zB = xyzB[(v*width + u)*3 + 2]

        # Get the rotated coordinates in A (world coordinates)
        x = xyzA[i*3+0];
        y = xyzA[i*3+1];
        z = xyzA[i*3+2];

        xA = matA[ 0]*x + matA[ 1]*y + matA[ 2]*z + matA[ 3];
        yA = matA[ 4]*x + matA[ 5]*y + matA[ 6]*z + matA[ 7];
        zA = matA[ 8]*x + matA[ 9]*y + matA[10]*z + matA[11];

        # Also rotate the normals
        x = normA[i*3+0];
        y = normA[i*3+1];
        z = normA[i*3+2];

        nxA = matA[ 0]*x + matA[ 1]*y + matA[ 2]*z;
        nyA = matA[ 4]*x + matA[ 5]*y + matA[ 6]*z;
        nzA = matA[ 8]*x + matA[ 9]*y + matA[10]*z;

        # The following is borrowed directly from
        # pr2_fasticp.cc provided by Szymon Rusinkiewicz
        # http://www.cs.princeton.edu/~smr/software/pr2/
        d = (xB - xA) * nxA + \
            (yB - yA) * nyA + \
            (zB - zA) * nzA;

        if d*d > dist2: continue
        err += d*d
        npairs += 1

        # Diagnostic: output u,v to check overlap
        uv[_blank*2+0] = u
        uv[_blank*2+1] = v
        

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


def _fast_icp(np.ndarray[np.float32_t, ndim=2, mode='c'] xyzA,
              np.ndarray[np.float32_t, ndim=2, mode='c'] normA,
              np.ndarray[np.float32_t, ndim=2, mode='c'] matA,
              np.ndarray[np.uint8_t, ndim=2, mode='c'] mask,
              np.ndarray[np.float32_t, ndim=3, mode='c'] xyzB,
              np.ndarray[np.float32_t, ndim=2, mode='c'] KKB,
              int samples, float dist2):
    height, width = mask.shape[0], mask.shape[1]
    assert xyzA.shape[1] == normA.shape[1] == xyzB.shape[2] == 3, 'xyz norm xyzB shape'
    assert matA.shape[0] == matA.shape[1] == 4, 'matA shape'
    assert KKB.shape[0] == KKB.shape[1] == 4, 'KKB shape'
    assert xyzA.shape[0] == normA.shape[0], 'xyzA shape'
    assert xyzB.shape[0] == height, 'xyzB shape'
    assert xyzB.shape[1] == width, 'xyzB shape'

    cdef np.ndarray[np.float32_t, ndim=2, mode='c'] matKKB = np.dot(KKB, matA)

    # Build an intermediate value UV coordinates matrix
    cdef np.ndarray[np.float32_t, ndim=2, mode='c'] uv = \
         np.zeros((samples, 2), dtype='f')

    # Build the accumulator matrices A (6x6) and B (6x1)
    cdef np.ndarray[np.float32_t, ndim=2, mode='c'] A = \
         np.zeros((6, 6), dtype='f')
    cdef np.ndarray[np.float32_t, ndim=1, mode='c'] B = \
         np.zeros((6,), dtype='f')

    #project_coords(<np.float32_t *> xyzA.data, xyzA.shape[0],
    #               <np.float32_t *> matKKB.data,
    #               <np.float32_t *> uv.data)
    err, npairs = _c_fast_icp(<np.float32_t *> xyzA.data,
                              <np.float32_t *> normA.data,
                              <np.float32_t *> matA.data,              
                              xyzA.shape[0],
                              <np.float32_t *> xyzB.data,
                              <np.float32_t *> matKKB.data,
                              <np.uint8_t *> mask.data,
                              <np.float32_t *> uv.data,                       
                              height, width,
                              samples, dist2,
                              <np.float32_t (*)[6]> A.data,
                              <np.float32_t *> B.data
                              )

    xvec = np.linalg.solve(A, B)

    RT = np.eye(4, dtype='f')
    RT[:3,:3] = transformations.euler_matrix(*xvec[:3])[:3,:3]
    RT[:3,3] = xvec[3:]
    
    return err, npairs, uv, RT


def fast_icp(range_image, point_model, samples=10000, dist=0.01):
    """Returns ICP between a range image and a point model.
    The range_image is held still, an altered 'point_model' is returned
    that's optimally aligned to the range image.

    The normals for point_model are used in the computation, normals
    for the range_image are ignored.

    Points are selected randomly from point_model and projected into the
    image. This is a terrible choice if:
       - only a portion of the point model is visible to the camera,
         e.g., very large object, narrow zoomed-in camera

    Args:
        range_image: a RangeImage
        point_model: a PointModel

    Returns:
        pnew: a PointModel identical to point_model except that
            pnew.RT has been modified so that pnew is aligned to
            range_image.
    """
    matKKB = np.dot(range_image.camera.KK, range_image.camera.RT)

    srand48(time(0))

    err, npairs, uv, RT  = _fast_icp(point_model.xyz,
                                     point_model.norm,
                                     point_model.RT,
                                     (range_image.depth < np.inf).astype('u1'),
                                     range_image.xyz,
                                     matKKB,
                                     samples, dist*dist)

    pnew = PointModel(point_model.xyz, point_model.norm,
                      #np.dot(np.linalg.inv(RT), point_model.RT))
                      np.dot(RT, point_model.RT))    

    return pnew, err, npairs, uv
