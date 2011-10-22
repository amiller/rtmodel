
#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>

struct Voxel {
    short int sd;
    short int w;
};

__global__ void incr_tsdf(Voxel *vox)
{
   for (int z = 0; z < 512; z++) {
      int idx = z*512*512 + blockIdx.x*512 + threadIdx.x;
      vox[idx].sd += threadIdx.x;
      vox[idx].w += blockIdx.x;
   }
}

int main(void) {
    Voxel *vox_gpu;
    Voxel *vox_cpu;

    vox_cpu = (Voxel *) malloc(512*512*512*4);
    cudaMalloc((void **) &vox_gpu, 512*512*512*4);
    dim3 dimBlock(512,1,1);
    dim3 dimGrid(512,1,1);

    int N = 10;

    cudaEvent_t e_start, e_stop;
    cudaEventCreate(&e_start);
    cudaEventCreate(&e_stop);
    cudaEventRecord(e_start);
    for (int i = 0; i < N; i++) {      
	incr_tsdf<<<dimGrid, dimBlock>>>(vox_gpu);
    }
    cudaEventRecord(e_stop);
    cudaEventSynchronize(e_stop);

    float ms;
    cudaEventElapsedTime(&ms, e_start, e_stop);

    cudaMemcpy(vox_cpu, vox_gpu, 512*512*512*4, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 20; i++) {
    	printf("[%03d] %d %d\n", i, (int) vox_cpu[i].sd, (int) vox_cpu[i].w);
    }

    printf("%d in %.1f (avg %.1f)\n", N, ms, ms/N);

    cudaFree(vox_gpu);
    free(vox_cpu);
    return 0;
}
