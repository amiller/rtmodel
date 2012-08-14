import numpy as np

lut = np.arange(5000).astype('f')
lut[1:] = 1000./lut[1:]
lut[0] = -1e8


def recip_depth_openni(depth):
    import scipy.weave
    assert depth.dtype == np.uint16
    output = np.empty(depth.shape,'f')
    N = np.prod(depth.shape)
    code = """
    int i;
    for (i = 0; i < (int)N; i++) {
      output[i] = lut[depth[i]];
    }
    """
    scipy.weave.inline(code, ['output','depth','lut','N'])
    return output


def projection():
    """
    Camera matrix for the depth aligned to rgb
    """
    fx = 528.0
    fy = 528.0
    cx = 320.0
    cy = 267.0

    mat = np.array([[fx, 0, -cx, 0],
                    [0, -fy, -cy, 0],
                    [0, 0, 0, 1],
                    [0, 0, -1., 0]])
    return np.ascontiguousarray(np.linalg.inv(mat).astype('f'))
