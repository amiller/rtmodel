import unittest
import numpy as np

from rtmodel import rangeimage
from rtmodel import rangeimage_speed
from rtmodel import mesh
from rtmodel import camera


def load_obj(name='blueghost'):
    #obj = mesh.load_random()
    obj = mesh.load(name)
    obj.RT[:3,3] = -obj.vertices[:,:3].mean(0)
    obj.RT[:3,3] += [0,0,-3.0]

    return obj


class TestRangeImage(unittest.TestCase):
    def setUp(self):
        self.obj = load_obj()

    def test_rangeimage(self):
        global pm1, pm2, rimg, obj
        obj = self.obj
        rimg = self.obj.range_render(camera.kinect_camera())
        pm1 = rimg.point_model_py()
        pm2 = rimg.point_model()        
        assert np.allclose(pm1.xyz, pm2.xyz, atol=1e-5)
        print 'Cython point_model matches point_model_py'


if __name__ == '__main__':
    unittest.main()
