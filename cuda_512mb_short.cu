/* Andrew Miller <amiller@dappervision.com>
 *
 * Cuda 512*512*512*4bytes test
 * 
 * According to the KinectFusion UIST 2011 paper, it's possible 
 * to do a sweep of 512^3 voxels, 32-bits each, in ~2ms on a GTX470.
 * 
 * This code is a simple benchmark accessing 512^3*2 short ints.
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

const int N_DATA = (512*512*512);
const int N_BYTES = (N_DATA*2);
const int N_GRID = 512*4;
const int N_BLOCK = 512;
const int N_CHUNK = 8;
const int N_FAN = N_DATA/N_GRID/N_BLOCK/N_CHUNK;
const int K = 13;
const int N_LOOPS = 10;

struct __align__(16) short8  {
    short s0, s1, s2, s3, s4, s5, s6, s7;
};

/*
  Each kernel processes several adjacent elements
  N_DATA = (N_GRID) * (N_FAN) * (N_BLOCK) * (N_CHUNK) = 512*512*512*2
 */

__global__ void incr_data3(short int *data) {
    // Outer loop skips by strides of N_BLOCK*N_CHUNK
    for (int i = 0; i < N_FAN; i++) {
	int idx = blockIdx.x*(N_FAN*N_BLOCK*N_CHUNK) + i*(N_BLOCK*N_CHUNK) + threadIdx.x*(N_CHUNK);
	int4 *dd = (int4 *) &data[idx];
	int4 d_ = *dd;
	short8 d = *((short8 *) &d_);
	d.s0 += K;
	d.s1 += K;
	d.s2 += K;
	d.s3 += K;
	d.s4 += K;
	d.s5 += K;
	d.s6 += K;
	d.s7 += K;
	*dd = *((int4 *) &d);
    }
}

/*
__global__ void incr_data1(short int *data) {
  // Outer loop skips by strides of N_BLOCK*N_CHUNK
  for (int i = 0; i < N_FAN; i++) {
    int idx = blockIdx.x*(N_FAN*N_BLOCK*N_CHUNK) + i*(N_BLOCK*N_CHUNK) + threadIdx.x*(N_CHUNK);

    // Inner loop processes 16 bytes (8 short ints) at once (a chunk)
    #pragma unroll
    for (int j = 0; j < N_CHUNK; j+=8, idx++) {
      short4 *d = (short4 *) data;
      d[(idx+j)/4].x += K;
      d[(idx+j)/4].y += K;
      d[(idx+j)/4].z += K;
      d[(idx+j)/4].w += K;
    }
  }
}



__global__ void copy_data1(short int *data, short int *out) {
  // Outer loop skips by strides of N_BLOCK*N_CHUNK
  for (int i = 0; i < N_FAN; i++) {
    int idx = blockIdx.x*(N_FAN*N_BLOCK*N_CHUNK) + i*(N_BLOCK*N_CHUNK) + threadIdx.x*(N_CHUNK);

    // Inner loop processes 16 bytes (8 short ints) at once (a chunk)
    #pragma unroll
    for (int j = 0; j < N_CHUNK; j+=8, idx++) {
      short4 *d = (short4 *) data;
      short4 *o = (short4 *) out;
      o[(idx+j)/4].x = d[(idx+j)/4].x;
      o[(idx+j)/4].y = d[(idx+j)/4].y;
      o[(idx+j)/4].z = d[(idx+j)/4].z;
      o[(idx+j)/4].w = d[(idx+j)/4].w;
    }
  }
}

__global__ void copy_data2(short int *data, short int *out) {
    // Outer loop skips by strides of N_BLOCK*N_CHUNK
    for (int i = 0; i < N_FAN; i++) {
	int idx = blockIdx.x*(N_FAN*N_BLOCK*N_CHUNK) + i*(N_BLOCK*N_CHUNK) + threadIdx.x*(N_CHUNK);
	int4 *dd = (int4 *) &data[idx];
	int4 *oo = (int4 *) &out[idx];
	short8 d = *((short8 *) dd);
	*oo = *((int4 *) &d);
    }
}

__global__ void incr_data2(short int *data) {
  // Outer loop skips by strides of N_BLOCK*N_CHUNK
  for (int i = 0; i < N_FAN; i++) {
    int idx = blockIdx.x*(N_FAN*N_BLOCK*N_CHUNK) + i*(N_BLOCK*N_CHUNK) + threadIdx.x*(N_CHUNK);

    // Inner loop processes 16 bytes (8 short ints) at once (a chunk)
    for (int j = 0; j < N_CHUNK; j+=8, idx++) {
      short8 *d = (short8 *) data;
      d[(idx+j)/8].s0 += K;
      d[(idx+j)/8].s1 += K;
      d[(idx+j)/8].s2 += K;
      d[(idx+j)/8].s3 += K;
      d[(idx+j)/8].s4 += K;
      d[(idx+j)/8].s5 += K;
      d[(idx+j)/8].s6 += K;
      d[(idx+j)/8].s7 += K;
    }
  }
}
*/
int main(void) {
  short int *data_gpu;
  short int *data_gpuA;
  short int *data_cpu;
  
  cudaMalloc((void **) &data_gpu, N_BYTES);
  cudaMalloc((void **) &data_gpuA, N_BYTES);
  data_cpu = (short int *) calloc(N_BYTES, 1);
  cudaMemcpy(data_gpu, data_cpu, N_BYTES, cudaMemcpyHostToDevice);

  dim3 dimBlock(N_BLOCK,1,1);
  dim3 dimGrid(N_GRID,1,1);    

  cudaEvent_t e_start, e_stop;
  cudaEventCreate(&e_start);
  cudaEventCreate(&e_stop);
  cudaEventRecord(e_start);

  // Run the kernel several times
  for (int i = 0; i < N_LOOPS; i++) {      
      incr_data3<<<dimGrid, dimBlock>>>(data_gpu);
      //incr_data3<<<dimGrid, dimBlock>>>(data_gpu, data_gpuA);
  }

  cudaEventRecord(e_stop);
  cudaEventSynchronize(e_stop);

  // Copy back to the host and check we have what we expect
  cudaMemcpy(data_cpu, data_gpu, N_BYTES, cudaMemcpyDeviceToHost);
  for (int i = 0; i < N_DATA; i++) {
      assert(data_cpu[i] == (short)N_LOOPS*K);
  }

  // Timing information
  float ms;
  cudaEventElapsedTime(&ms, e_start, e_stop);  
  printf("%d sweeps of %.1f megabytes in %.1fms (avg %.1fms)\n", 
	 N_LOOPS, N_BYTES/1000.0/1000.0, ms, ms/N_LOOPS);

  cudaFree(data_gpu);
  free(data_cpu);
  return 0;
}
