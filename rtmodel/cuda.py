import pycuda.gpuarray as gpuarray
import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel

vox_gpu = gpuarray.empty((512,512,512,2), dtype=np.int16)

def free():
    global vox_gpu
    if 'vox_gpu' in globals():
        del vox_gpu

mod = SourceModule("""
struct Voxel {
  short int d;
  short int w;
};

__global__ void zero_tsdf(Voxel *vox)
{
   int idx = blockIdx.x*512*512 + blockIdx.y*512 + threadIdx.x;
   vox[idx].d += threadIdx.x;
   vox[idx].w += threadIdx.y;
}
/*
__global__ void zero_tsdf_2(short int vox[512][512][512][2]) {
   vox[threadIdx.z][threadIdx.y][threadIdx.x][0] = threadIdx.x;
   vox[threadIdx.z][threadIdx.y][threadIdx.x][1] = threadIdx.y;
}*/
""")

_zero_tsdf_3 = ElementwiseKernel(
    "short int *vox",
    "vox[i] += i;",
    "zero_tsdf_3")

_zero_tsdf = mod.get_function('zero_tsdf')


def zero_tsdf():
    _zero_tsdf(vox_gpu, block=(512,1,1), grid=(512,512))
    pycuda.autoinit.context.synchronize()

def zero_tsdf3():
    _zero_tsdf_3(vox_gpu)
    pycuda.autoinit.context.synchronize()
