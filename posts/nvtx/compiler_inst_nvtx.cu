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
