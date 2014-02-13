#pragma once

#include "fake_atomic.h"


__global__ void device_reduce_atomic_kernel(int *in, int* out, int N) {
  int sum=int(0);
  for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<N;i+=blockDim.x*gridDim.x) {
    sum+=in[i];
  }
  atomicAdd(out,sum);
}

void device_reduce_atomic(int *in, int* out, int N) {
  int threads=256;
  int blocks=min((N+threads-1)/threads,2048);

  cudaMemsetAsync(out,0,sizeof(int));
  device_reduce_atomic_kernel<<<blocks,threads>>>(in,out,N); 
}

__global__ void device_reduce_atomic_kernel_vector2(int *in, int* out, int N) {
  int sum=0;
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  for(int i=idx;i<N/2;i+=blockDim.x*gridDim.x) {
    int2 val=reinterpret_cast<int2*>(in)[i];
    sum+=val.x+val.y;
  }
  int i=idx+N/2*2;
  if(i<N) 
    sum+=in[i];
  
  atomicAdd(out,sum);
}

void device_reduce_atomic_vector2(int *in, int* out, int N) {
  int threads=256;
  int blocks=min((N/2+threads-1)/threads,2048);

  cudaMemsetAsync(out,0,sizeof(int));
  device_reduce_atomic_kernel_vector2<<<blocks,threads>>>(in,out,N); 
}

__global__ void device_reduce_atomic_kernel_vector4(int *in, int* out, int N) {
  int sum=0;
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  for(int i=idx;i<N/4;i+=blockDim.x*gridDim.x) {
    int4 val=reinterpret_cast<int4*>(in)[i];
    sum+=(val.x+val.y)+(val.z+val.w);
  }
  int i=idx+N/4*4;
  if(i<N) 
    sum+=in[i];
  
  atomicAdd(out,sum);
}

void device_reduce_atomic_vector4(int *in, int* out, int N) {
  int threads=256;
  int blocks=min((N/4+threads-1)/threads,2048);

  cudaMemsetAsync(out,0,sizeof(int));
  device_reduce_atomic_kernel_vector4<<<blocks,threads>>>(in,out,N); 
}
