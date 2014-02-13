#pragma once

//#define __shfl_down fake_shfl_down
#define MAX_BLOCK 512
__inline__ __device__
int fake_shfl_down(int val, int offset, int width=32) {
  static __shared__ int shared[MAX_BLOCK];
  int lane=threadIdx.x%32;

  shared[threadIdx.x]=val;
  __syncthreads();

  val = (lane+offset<width) ? shared[threadIdx.x+offset] : 0;
  __syncthreads();

  return val;
}
