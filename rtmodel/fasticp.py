import pointmodel
import rangeimage


def FastICP(depth_img, model, cam):
    """Performs Fast ICP between a point model and a range image.

    Returns:
       dict containing:
        'M':
            a 4x4 matrix such that M * point.RT * point.xyz
            puts the point model in alignment with the range image
        'err':
            reprojection error, sum squared meters
    """
    pass
