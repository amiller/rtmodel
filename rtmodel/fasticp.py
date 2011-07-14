import numpy as np
import pointmodel
import rangeimage
from _fasticp import _fasticp
import transformations


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
    global A, B, uv, mask, npairs, matKKBtoA, matB


    # The computation must actually take place in the coordinate system
    # of range_image.camera.RT
    camRTinv = np.linalg.inv(range_image.camera.RT)
    matBtoA = np.dot(camRTinv, point_model.RT)
    matBtoA = np.ascontiguousarray(matBtoA)

    matKKBtoA = np.dot(range_image.camera.KK, matBtoA)

    mask = (range_image.depth < np.inf).astype('u1')
    err, npairs, uv, A, B = _fasticp(range_image.xyz,
                                     range_image.normals,
                                     mask,
                                     point_model.xyz,
                                     matBtoA,
                                     matKKBtoA,
                                     samples, dist*dist)

    xvec = np.linalg.solve(A, B)
    RT = np.eye(4, dtype='f')
    RT[:3,:3] = transformations.euler_matrix(*-xvec[:3])[:3,:3]
    RT[:3,3] = -xvec[3:]

    # camera.RT * RT * camRTinv * point_model.RT
    RT = np.dot(range_image.camera.RT, np.dot(RT, np.dot(camRTinv, point_model.RT)))
    pnew = pointmodel.PointModel(point_model.xyz, point_model.normals, np.ascontiguousarray(RT))

    return pnew, err, npairs, uv
