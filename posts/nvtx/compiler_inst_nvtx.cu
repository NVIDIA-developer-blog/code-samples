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
	for (int i=0; i<n; ++i)
	{
		x[i] = i;
	}
}

void init_data(int n, double* x, double* x_d, double* y_d)
{
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
}

void daxpy(int n, double a, double* x_d, double* y_d)
{
	daxpy_kernel<<<ceil(n/256),256>>>(n,a,x_d,y_d);
	cudaDeviceSynchronize();
}

void check_results( int n, double correctvalue, double* x_d )
{
	check_results_kernel<<<ceil(n/256),256>>>(n,correctvalue,x_d);
}

void run_test(int n)
{
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
}

int main()
{
	int n = 1<<22;
	run_test(n);
	return 0;
}
