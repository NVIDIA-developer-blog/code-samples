// Copyright (c) 1993-2016, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#include <cstdio>
#include <cuda_fp16.h>
#include <assert.h>
#include "fp16_conversion.h"

// This is a simple example of using FP16 types and arithmetic on
// GPUs that support it. The code computes an AXPY (A * X + Y) operation
// on half-precision (FP16) vectors (HAXPY).

// Convenience function for checking CUDA runtime API results
// can be wrapped around any runtime API call. No-op in release builds.
inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

__global__
void haxpy(int n, half a, const half *x, half *y)
{
	int start = threadIdx.x + blockDim.x * blockIdx.x;
	int stride = blockDim.x * gridDim.x;

#if __CUDA_ARCH__ >= 530
  int n2 = n/2;
	half2 *x2 = (half2*)x, *y2 = (half2*)y;

	for (int i = start; i < n2; i+= stride) 
		y2[i] = __hfma2(__halves2half2(a, a), x2[i], y2[i]);

	// first thread handles singleton for odd arrays
  if (start == 0 && (n%2))
  	y[n-1] = __hfma(a, x[n-1], y[n-1]);   

#else
  for (int i = start; i < n; i+= stride) {
    y[i] = __float2half(__half2float(a) * __half2float(x[i]) 
    		                                + __half2float(y[i]));
  }
#endif
}

int main(void) {
	const int n = 100;

	const half a = approx_float_to_half(2.0f);

	half *x, *y;
	checkCuda(cudaMallocManaged(&x, n * sizeof(half)));
	checkCuda(cudaMallocManaged(&y, n * sizeof(half)));
	
	for (int i = 0; i < n; i++) {
		x[i] = approx_float_to_half(1.0f);
		y[i] = approx_float_to_half((float)i);
	}

	const int blockSize = 256;
	const int nBlocks = (n + blockSize - 1) / blockSize;

	haxpy<<<nBlocks, blockSize>>>(n, a, x, y);

  // must wait for kernel to finish before CPU accesses
  checkCuda(cudaDeviceSynchronize());
    
  for (int i = 0; i < n; i++)
  	printf("%f\n", half_to_float(y[i]));

  return 0;
}

