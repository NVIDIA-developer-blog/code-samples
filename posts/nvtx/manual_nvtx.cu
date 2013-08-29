#include <cstdio>

#ifdef USE_NVTX
#include "nvToolsExt.h"

const uint32_t colors[] = { 0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff, 0x0000ffff, 0x00ff0000, 0x00ffffff };
const int num_colors = sizeof(colors)/sizeof(uint32_t);

#define START_RANGE(name,cid) { \
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
#define END_RANGE nvtxRangePop();
#else
#define START_RANGE(name,cid)
#define END_RANGE
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
	START_RANGE("init_host_data",1)
	for (int i=0; i<n; ++i)
	{
		x[i] = i;
	}
	END_RANGE
}

void init_data(int n, double* x, double* x_d, double* y_d)
{
	START_RANGE("init_data",2)
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
	END_RANGE
}

void daxpy(int n, double a, double* x_d, double* y_d)
{
	START_RANGE("daxpy",3)
	daxpy_kernel<<<ceil(n/256),256>>>(n,a,x_d,y_d);
	cudaDeviceSynchronize();
	END_RANGE
}

void check_results( int n, double correctvalue, double* x_d )
{
	START_RANGE("check_results",4)
	check_results_kernel<<<ceil(n/256),256>>>(n,correctvalue,x_d);
	END_RANGE
}

void run_test(int n)
{
	START_RANGE("run_test",0)
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
	END_RANGE
}

int main()
{
	int n = 1<<22;
	run_test(n);
	return 0;
}
