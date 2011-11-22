import unittest
from rtmodel import volume
from rtmodel import cuda
reload(cuda)

class T(unittest.TestCase):
    def _test_N_reload(self, N):
        # Try to load 512MB of data, then reload the module. Check
        # that reloading the module replaces the functions

        def check():
            v = cuda.CudaVolume(512)
            return v.test_tsdf()

        assert check() is None
        cuda.test_tsdf = lambda v: 'test'
        assert check() == 'test'
        reload(cuda)
        assert check() is None

    def test_512_reload(self):
        return self._test_N_reload(512)

    def test_256_reload(self):
        return self._test_N_reload(256)

    def test_test_tsdf(self):
        v = cuda.CudaVolume(128)
        v.test_tsdf()

    def volume(self):
        pass

if __name__ == '__main__':
    unittest.main()
