import numpy as np
import calibkinect

class Camera(object):

    def __init__(self, KK, RT=np.eye(4, dtype='f')):
        """
        Args:
            KK: 4x4 numpy matrix ('f')
                intrinsic parameters, origin remains at 0
            RT: 4x4 numpy matrix ('f')
                rotation and translation
        """
        self.KK = KK
        self.RT = RT


def kinect_camera():
    return Camera(calibkinect.projection())
