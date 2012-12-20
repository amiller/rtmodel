import numpy as np
import pointmodel
import rangeimage
from _fasticp import _fasticp
import transformations


def fast_icp_multi(range_images, point_models, dist=0.01, normthresh=0.7):
    global A, B, uv, mask, npairs, matKK

    # For now, we assume that the xyz/normal data for both the
    # point models and the range images are already in the same
    # coordinate space.
    within_eps = lambda a, b: np.all(np.abs(a-b)) < 1e-5
    for rimg,pm in zip(range_images, point_models):
        assert within_eps(rimg.RT, pm.RT), "Point model and range image data \
            are assumed to be in the same coordinate space."

    # Initialize accumulators
    A = np.zeros((6,6))
    B = np.zeros((6,))
    uv = []
    sse = 0.
    npairs = 0

    # Accumulate for each pair of range images and point clouds
    for rimg,pm in zip(range_images, point_models):
        KK = np.linalg.inv(np.dot(rimg.camera.RT, rimg.camera.KK))
        sse_, npairs_, uv_, A_, B_ = _fasticp(rimg.xyz,
                                              rimg.normals,
                                              rimg.mask.astype('u1'),
                                              pm.xyz,
                                              pm.normals,
                                              np.ascontiguousarray(KK),
                                              dist*dist, normthresh)
        npairs += npairs_
        sse += sse_
        uv.append(uv_)
        A += A_
        B += B_

    # Fill out bottom triangle
    for i in range(6):
        for j in range(i):
            A[i,j] = A[j,i]

    xvec = np.linalg.solve(A, B)
    RT = np.eye(4, dtype='f')
    RT[:3,:3] = transformations.euler_matrix(*-xvec[:3])[:3,:3]
    RT[:3,3] = -xvec[3:]

    return RT, sse, npairs, uv


def fast_icp(range_image, point_model, rate=0.001, dist=0.01):
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
                                     rate, dist*dist)

    xvec = np.linalg.solve(A, B)
    RT = np.eye(4, dtype='f')
    RT[:3,:3] = transformations.euler_matrix(*-xvec[:3])[:3,:3]
    RT[:3,3] = -xvec[3:]

    # camera.RT * RT * camRTinv * point_model.RT
    RT = np.dot(range_image.camera.RT, np.dot(RT, np.dot(camRTinv, point_model.RT)))
    pnew = pointmodel.PointModel(point_model.xyz, point_model.normals, np.ascontiguousarray(RT))
    pnew.mask = mask

    return pnew, err, npairs, uv
