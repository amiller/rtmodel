/* Andrew Miller <amiller@dappervision.com>
 *
 * Cuda 512*512*512*4bytes test
 * 
 * According to the KinectFusion UIST 2011 paper, it's possible 
 * to do a sweep of 512^3 voxels, 32-bits each, in ~2ms on a GTX470.
 * 
 * This code is a simple benchmark accessing 512^3 voxels. Each
 * voxel has two 16-bit components. In this benchmark kernel, we
 * simply increment these values by a constant K. More than anything
 * it's a test of the memory bandwidth.
 *
 * On my GTX470 card, this kernel takes 10.7ms instead of ~2ms. Is there
 * a faster way to do this?
 *
 * Citation: http://dl.acm.org/citation.cfm?id=2047270
 * Public gdocs pdf link: http://tinyurl.com/6xlznbx
 */

#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>
#include <assert.h>

struct Voxel {
    short int sd;
    short int w;
};

const int N_BYTES = (512*512*512*4);
const int N_LOOPS = 10;
const int K = 7;

__global__ void incr_tsdf(Voxel *vox, Voxel *out)
{
    int idx = blockIdx.x*512 + threadIdx.x;
    for (int z = 0; z < 512; z++) {
	out[idx].sd = vox[idx].sd + K;
	out[idx].w = vox[idx].w += K;
	idx += 512*512;
    }
}

int main(void) {
    Voxel *vox_gpu;
    Voxel *vox_gpuA;
    Voxel *vox_cpu;

    cudaMalloc((void **) &vox_gpu, N_BYTES);
    cudaMalloc((void **) &vox_gpuA, N_BYTES);
    vox_cpu = (Voxel *) calloc(N_BYTES, 1);
    cudaMemcpy(vox_gpu, vox_cpu, N_BYTES, cudaMemcpyHostToDevice);

    dim3 dimBlock(512,1,1);
    dim3 dimGrid(512,1,1);

    cudaEvent_t e_start, e_stop;
    cudaEventCreate(&e_start);
    cudaEventCreate(&e_stop);
    cudaEventRecord(e_start);
    for (int i = 0; i < N_LOOPS; i++) {      
	incr_tsdf<<<dimGrid, dimBlock>>>(vox_gpu, vox_gpuA);
    }
    cudaEventRecord(e_stop);
    cudaEventSynchronize(e_stop);

    float ms;
    cudaEventElapsedTime(&ms, e_start, e_stop);

    // Copy back to the host and check we have what we expect
    cudaMemcpy(vox_cpu, vox_gpu, N_BYTES, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 512; i++) {
    	for (int j = 0; j < 512; j++) {
	    for (int k = 0; k < 512; k++) {
	    	int idx = i*512*512 + j*512 + k;
	    	assert(vox_cpu[idx].sd == (short)N_LOOPS*K);
	    	assert(vox_cpu[idx].w == (short)N_LOOPS*K);
	    }
	}
    }

    printf("%d sweeps of %.1f megavoxels in %.1fms (avg %.1fms)\n", 
        N_LOOPS, N_BYTES/4.0/1000.0/1000.0, ms, ms/N_LOOPS);

    cudaFree(vox_gpu);
    free(vox_cpu);
    return 0;
}
