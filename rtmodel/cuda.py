import cuda_init
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import pycuda
from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel
import numpy as np


mod = SourceModule("""
struct Voxel {
  short int d;
  short int w;
};

__device__ void _test_tsdf(Voxel &vox, const int X, const int Y, const int Z) {
    vox.d += X;
    vox.w += Y;
}

__global__ void test_tsdf(Voxel *vox, const int N) {
    int X = blockIdx.x;
    int Y = blockIdx.y;
    int Z = threadIdx.x;
    int idx = blockIdx.x*N*N + blockIdx.y*N + threadIdx.x;
    _test_tsdf(vox[idx], X, Y, Z);
}

//const int WIDTH = 640;
//const int HEIGHT = 480;

texture<float, 2, cudaReadModeElementType> depth_tex;

__device__ void _init_tsdf(Voxel &vox,
                           const float X, const float Y, const float Z,
                           const float m[16],
                           const float MAX_D) {
    float x = X*m[ 0] + Y*m[ 1] + Z*m[ 2] + m[ 3];
    float y = X*m[ 4] + Y*m[ 5] + Z*m[ 6] + m[ 7];
    float z = X*m[ 8] + Y*m[ 9] + Z*m[10] + m[11];
    float w = X*m[12] + Y*m[13] + Z*m[14] + m[15];

    w = w == 0 ? 0 : 1./w;
    x = x * w;
    y = y * w;
    z = z * w;

    float dref = 1./z;

    int ix = (int) x;
    int iy = (int) y;

    /*if (ix < 0) return;
    if (ix > WIDTH) return;
    if (iy < 0) return;
    if (iy >= HEIGHT) return;*/
    /*ix = min(max(0, ix), WIDTH);
    iy = min(max(0, iy), HEIGHT);*/

    float d = tex2D(depth_tex, ix, iy);

    // Clamp and apply weighting function
    float sd = d - dref;
    float sw = 1;

    if (sd < -MAX_D) {
        sw = 0;
        sd = -MAX_D;
    } else if (sd > MAX_D) {
        sw = 0;
        sd = MAX_D;
    }
    sd = sd * 32767 / MAX_D;
    sw = sw * 32767;

    vox.d = (short int) sd;
    vox.w = (short int) sw;
}

__constant__ float mat[16];

__global__ void init_tsdf(Voxel *vox,
                          const float MAX_D,
                          const int N) {
    int idx = blockIdx.x*N*N + blockIdx.y*N + threadIdx.x;
    Voxel &v = vox[idx];
    _init_tsdf(v, blockIdx.x, blockIdx.y, threadIdx.x, mat, MAX_D);
}

const float NEAR = 0.5;
const float FAR = 10.0;
const float D_NEAR = 1000./NEAR;
__device__ void _raycast_tsdf(float4 &synth,
                              const Voxel *vox, const int N,
                              const float m[16],
                              const float cx, float cy, float cz,
                              const float X, const float Y,
                              const float SKIP_A, const float SKIP_B) {

    // Decide where to start the raycast
    float dt = NEAR;
    float Z = D_NEAR;

    float x = X*m[ 0] + Y*m[ 1] + Z*m[ 2] + m[ 3];
    float y = X*m[ 4] + Y*m[ 5] + Z*m[ 6] + m[ 7];
    float z = X*m[ 8] + Y*m[ 9] + Z*m[10] + m[11];
    float w = X*m[12] + Y*m[13] + Z*m[14] + m[15];

    w = 1./w;
    x *= w;
    y *= w;
    z *= w;

    float dx = x-cx;
    float dy = y-cy;
    float dz = z-cz;
    float _dn = SKIP_B/sqrtf(dx*dx + dy*dy + dz*dz);
    dx *= _dn;
    dy *= _dn;
    dz *= _dn;
    
    // Advance forward
    // increment dt a bit
    #define _IN_BOUNDS() ((x >= 0) && (x < N-1) && \
                          (y >= 0) && (y < N-1) && \
                          (z >= 0) && (z < N-1))
    #define _IDX(x,y,z) (((int)(x))*N*N + ((int)(y))*N + ((int)(z)))

    // First Pass: try to enter the surface
    Voxel v, v0;
    float dt0;
    while (!_IN_BOUNDS() && dt < FAR) {    
        dt += SKIP_A;
        x += dx;
        y += dy;
        z += dz;
    }
    
    if (!_IN_BOUNDS()) {
        synth.x = synth.y = synth.z = synth.w = 1;
        return;
    }

    // Second Pass: try to find a zero crossing
    while (_IN_BOUNDS() && dt < FAR) {
        v = vox[_IDX(x,y,z)];
        // Zero crossing detected
        if (v.d < 0) {
            break;
        }
        v0 = v;
        dt0 = dt;
        dt += SKIP_A;
        x += dx;
        y += dy;
        z += dz;       
    }

    if ((v0.d < 0) || (v.d > 0)) {
        synth.x = synth.y = synth.z = synth.w = 0;
        return;
    }

    synth.x = (dt0 * -v.d + dt * v0.d) / (v0.d - v.d);
    // normals computation
    // sample the grid at this point
    // and in dx,dy,dz directions
    {
        // Wind back to the actual intersection point
        float f = -(dt - synth.x) * (SKIP_B/SKIP_A);
        x += f * dx;
        y += f * dy;
        z += f * dz;
        float w  = (float) vox[_IDX(x,y,z)].d;
        float wx = vox[_IDX(x+1,y,z)].d - w;
        float wy = vox[_IDX(x,y+1,z)].d - w;
        float wz = vox[_IDX(x,y,z+1)].d - w;
        float nn = 1./sqrtf(wx*wx + wy*wy + wz*wz);
        wx *= nn;
        wy *= nn;
        wz *= nn;
        synth.y = wx; synth.z = wy; synth.w = wz;
    }
}

__global__ void raycast_tsdf(float4 *synth,
                             const Voxel *vox, const int N,
                             const int width, const int height,
                             const float cx, const float cy, const float cz,
                             const float SKIP_A, const float SKIP_B) {
    // Loop through height, width
    int x = blockIdx.x * 4 + threadIdx.x;
    int y = blockIdx.y * 4 + threadIdx.y;
    int idx = y * width + x;
    _raycast_tsdf(synth[idx],
                  vox, N, mat,
                  cx, cy, cz,
                  x, y, SKIP_A, SKIP_B);
}
""", options=['-use_fast_math'])

class Kernel():
    depth_gpu = gpuarray.empty((480,640), dtype=np.float32)
    synth_gpu = gpuarray.empty((480,640,4), dtype=np.float32)
    # Get a pagelocked reference to the data
    depth_cpu = depth_gpu.get(pagelocked=True)
    synth_cpu = synth_gpu.get(pagelocked=True)
    descr = pycuda.driver.ArrayDescriptor()
    descr.format = pycuda.driver.array_format.FLOAT
    descr.height = 480
    descr.width = 640
    descr.num_channels = 1
    depth_tex = mod.get_texref('depth_tex')
    depth_tex.set_filter_mode(pycuda.driver.filter_mode.POINT)
    depth_tex.set_address_2d(depth_gpu.gpudata, descr, 4*descr.width)
    #depth_tex.set_format(pycuda.driver.array_format.FLOAT, 1)
    test_tsdf = mod.get_function('test_tsdf')
    init_tsdf = mod.get_function('init_tsdf')
    raycast_tsdf = mod.get_function('raycast_tsdf')
    m_ptr, _  = mod.get_global('mat')

kernel = Kernel()


class CudaVolume(object):

    def __init__(self, N=128):
        self.vox_gpu = gpuarray.empty((N,N,N,2), dtype=np.int16)
        self.N = N

    def __getattr__(self, item):
        try:
            return _method_dict[item]
        except KeyError:
            raise AttributeError, item

_method_dict = {}
def method(target):
    def wrapper(self, *args, **kwargs):
        return globals()[target.__name__](self, *args, **kwargs)
    setattr(CudaVolume, target.__name__, wrapper)
    _method_dict[target.__name__] = target
    return target
    
@method
def test_tsdf(volume):
    N = volume.N
    kernel.test_tsdf(volume.vox_gpu, np.int32(N),
                     block=(N,1,1), grid=(N,N))
    pycuda.autoinit.context.synchronize()

@method
def load_depth_A(depth):
    assert depth.dtype == np.float32
    assert depth.shape == (480,640)
    strm = pycuda.driver.Stream()
    kernel.depth_cpu[:,:] = depth
    pycuda.driver.memcpy_htod_async(kernel.depth_gpu.gpudata, kernel.depth_cpu, strm)
    strm.synchronize()

@method
def load_depth_B(depth):
    assert depth.dtype == np.float32
    assert depth.shape == (480,640)
    strm = pycuda.driver.Stream()    
    kernel.depth_gpu.set(depth)
    strm.synchronize()

@method
def init_tsdf(volume, depth, mat, MAX_D=0.02):
    N = volume.N
    MAX_D = np.float32(MAX_D)
    assert mat.dtype == np.float32
    mat = mat.flatten()
    assert mat.shape == (16,)
    assert mat.flags['C_CONTIGUOUS']
    assert depth.dtype == np.float32
    assert depth.shape == (480,640)
    
    strm = pycuda.driver.Stream()
    pycuda.driver.memcpy_htod(kernel.m_ptr, mat)
    load_depth_B(depth)

    kernel.init_tsdf(volume.vox_gpu, MAX_D,
                     np.int32(N),
                     grid=(N,N), block=(N,1,1))
    strm.synchronize()

@method
def raycast_tsdf(volume, mat, c, SKIP_A, SKIP_B):
    """
__global__ void raycast_tsdf(float4 *synth,
                             const Voxel *vox, const int N,
                             const int width, const int height,
                             const float cx, const float cy, const float cz,
                             const float SKIP_A, const float SKIP_B) {
"""
    N = volume.N
    SKIP_A = np.float32(SKIP_A)
    SKIP_B = np.float32(SKIP_B)
    assert mat.dtype == np.float32
    mat = mat.flatten()
    assert mat.shape == (16,)
    assert mat.flags['C_CONTIGUOUS']
    cx, cy, cz = map(np.float32, c)
    pycuda.driver.memcpy_htod(kernel.m_ptr, mat)
    kernel.raycast_tsdf(kernel.synth_gpu,
                        volume.vox_gpu, np.int32(N),
                        np.int32(640), np.int32(480),
                        cx, cy, cz,
                        np.float32(SKIP_A), np.float32(SKIP_B),
                        grid=(640/4,480/4), block=(4,4,1))
    pycuda.autoinit.context.synchronize()
    synth = kernel.synth_gpu.get()
    volume.c_data = synth
    volume.c_depth = synth[:,:,0]
    volume.c_norm = synth[:,:,1:4]
