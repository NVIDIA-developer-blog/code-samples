/* Copyright (c) 2014, NVIDIA CORPORATION. All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <cstdio>

#ifdef USE_NVTX
#include "nvToolsExt.h"

const uint32_t colors[] = { 0xff00ff00, 0xff0000ff, 0xffffff00, 0xffff00ff, 0xff00ffff, 0xffff0000, 0xffffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define PUSH_RANGE(name,cid) { \
	int color_id = cid; \
	color_id = color_id%num_colors;\
	nvtxEventAttributes_t eventAttrib = {0}; \
	eventAttrib.version = NVTX_VERSION; \
	eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE; \
	eventAttrib.colorType = NVTX_COLOR_ARGB; \
	eventAttrib.color = colors[color_id]; \
	eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
	eventAttrib.message.ascii = name; \
	nvtxRangePushEx(&eventAttrib); \
}
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name,cid)
#define POP_RANGE
#endif

__global__ void init_data_kernel( int n, double* x)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if ( i < n )
	{
		x[i] = n - i;
	}
}


__global__ void daxpy_kernel(int n, double a, double * x, double * y)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
	{
		y[i] = a*x[i] + y[i];
	}
}

__global__ void check_results_kernel( int n, double correctvalue, double * x )
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if (i < n)
	{
		if ( x[i] != correctvalue )
		{
			printf("ERROR at index = %d, expected = %f, actual: %f\n",i,correctvalue,x[i]);
		}
	}
}

void init_host_data( int n, double * x )
{
	PUSH_RANGE("init_host_data",1)
	for (int i=0; i<n; ++i)
	{
		x[i] = i;
	}
	POP_RANGE
}

void init_data(int n, double* x, double* x_d, double* y_d)
{
	PUSH_RANGE("init_data",2)
	cudaStream_t copy_stream;
	cudaStream_t compute_stream;
	cudaStreamCreate(&copy_stream);
	cudaStreamCreate(&compute_stream);

	cudaMemcpyAsync( x_d, x, n*sizeof(double), cudaMemcpyDefault, copy_stream );
	init_data_kernel<<<ceil(n/256),256,0,compute_stream>>>(n, y_d);

	cudaStreamSynchronize(copy_stream);
	cudaStreamSynchronize(compute_stream);

	cudaStreamDestroy(compute_stream);
	cudaStreamDestroy(copy_stream);
	POP_RANGE
}

void daxpy(int n, double a, double* x_d, double* y_d)
{
	PUSH_RANGE("daxpy",3)
	daxpy_kernel<<<ceil(n/256),256>>>(n,a,x_d,y_d);
	cudaDeviceSynchronize();
	POP_RANGE
}

void check_results( int n, double correctvalue, double* x_d )
{
	PUSH_RANGE("check_results",4)
	check_results_kernel<<<ceil(n/256),256>>>(n,correctvalue,x_d);
	POP_RANGE
}

void run_test(int n)
{
	PUSH_RANGE("run_test",0)
	double* x;
	double* x_d;
	double* y_d;
	cudaSetDevice(0);
	cudaMallocHost((void**) &x, n*sizeof(double));
	cudaMalloc((void**)&x_d,n*sizeof(double));
	cudaMalloc((void**)&y_d,n*sizeof(double));

	init_host_data(n, x);

	init_data(n,x,x_d,y_d);

	daxpy(n,1.0,x_d,y_d);

	check_results(n, n, y_d);

	cudaFree(y_d);
	cudaFree(x_d);
	cudaFreeHost(x);
	cudaDeviceSynchronize();
	POP_RANGE
}

int main()
{
	int n = 1<<22;
	run_test(n);
	return 0;
}
