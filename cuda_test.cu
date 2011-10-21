#include <stdio.h>
#include <cuda.h>
#include "timer.h"

struct Voxel {
    short int d;
    short int w;
};

__global__ void zero_tsdf(Voxel *vox)
{
   int idx = blockIdx.x*512*512 + blockIdx.y*512 + threadIdx.z;
   vox[idx].d = threadIdx.z;
   vox[idx].w = threadIdx.y;
}

int main(void) {
    Voxel *vox_gpu;
    Voxel *vox_cpu;
    vox_cpu = (Voxel *) malloc(512*512*512*4);
    cudaMalloc((void **) &vox_gpu, 512*512*512*4);
    dim3 dimBlock(512,1,1);
    dim3 dimGrid(512,512,1);
    cpu_timer timer;

    timer.start();
    int N = 10000;

    cudaEvent_t event1;
    cudaEventCreate(&event1);
    cudaEventRecord(event1);
    for (int i = 0; i < N; i++) {       
	zero_tsdf<<<dimGrid, dimBlock>>>(vox_gpu);
    }
    cudaEventSynchronize(event1);
    timer.stop();

    cudaMemcpy(vox_cpu, vox_gpu, 512*512*512*4, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 20; i++) {
    	printf("[%03d] %d %d\n", i, (int) vox_gpu[i].d, (int) vox_gpu[i].w);
    }

    printf("%d in %.1f (avg %.1f)\n", N, timer.elapsed_ms(), timer.elapsed_ms()/N);
    cudaFree(vox_gpu);
    return 0;
}
