/* Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
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


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>

#include "common.h"

#define TILE_DIM 32
#define BLOCK_ROWS 8
#define USE_OPTIMIZED_TRANSPOSE 0

////////////////////////////////////////////////////////////////////////////////
// Custom Kernels Implementations
////////////////////////////////////////////////////////////////////////////////
__global__ void ConvertInputR(
        const char * __restrict__ dataIn, 
        cufftReal * __restrict__ dataOut) 
{
    const int numThreads = blockDim.x * gridDim.x;
    const int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    
    for(size_t offset = threadId; offset < INPUT_SIGNAL_SIZE * BATCH_SIZE; offset += numThreads) {
        char element = dataIn[offset];
        dataOut[offset] = (cufftReal)((float)element/127.0f);
    }
}

__global__ void ConvolveAndStoreTransposedC_Basic(
    const cufftComplex * __restrict__ dataIn, 
    cufftComplex * __restrict__ dataOut,
    const cufftComplex * __restrict__ filter)
{

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int yBase = blockIdx.y * TILE_DIM + threadIdx.y;
    
    if(x < COMPLEX_SIGNAL_SIZE) {
        for(int j = 0; j < TILE_DIM; j+= BLOCK_ROWS) {
            int y = yBase + j;
            if(y >= BATCH_SIZE) break;
            cufftComplex value = ComplexMul(dataIn[y * COMPLEX_SIGNAL_SIZE + x], filter[x]);
            dataOut[x*BATCH_SIZE + y] = value;
        }
    }
}

__global__ void ConvolveAndStoreTransposedC_Optimized(
    const cufftComplex * __restrict__ dataIn, 
    cufftComplex * __restrict__ dataOut,
    const cufftComplex * __restrict__ filter)
{
    __shared__ cufftComplex tile[TILE_DIM][TILE_DIM+1];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int yBase = blockIdx.y * TILE_DIM + threadIdx.y;
    
    if(x < COMPLEX_SIGNAL_SIZE) {
        for(int j = 0; j < TILE_DIM; j+= BLOCK_ROWS) {
            int y = yBase + j;
            if(y >= BATCH_SIZE) break;
            cufftComplex value = ComplexMul(dataIn[y * COMPLEX_SIGNAL_SIZE + x], filter[x]);
            tile[threadIdx.y + j][threadIdx.x] = value;
        }
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    yBase = blockIdx.x * TILE_DIM + threadIdx.y;

    if(x < BATCH_SIZE) {
        for(int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
            int y = yBase + j;
            if(y >= COMPLEX_SIGNAL_SIZE) break;
            dataOut[y * BATCH_SIZE + x] = tile[threadIdx.x][threadIdx.y + j];
        }  
    }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, const char **argv)
{
    struct cudaDeviceProp properties;
    int device = argc > 1 ? atoi(argv[1]) : 0;

    checkCudaErrors(cudaGetDevice(&device));
    checkCudaErrors(cudaGetDeviceProperties(&properties, device));
    if( !(properties.major >= 2) ) {
        printf("This sample requires CUDA architecture SM2.0 or higher\n");
        exit(EXIT_FAILURE);
    }

    // Allocate and initialize memory
    printf("Preparing input: %dx%d\n", BATCH_SIZE, INPUT_SIGNAL_SIZE);
    char *_8bit_signal;
    cufftReal *tmp_result1;
    cufftComplex *tmp_result2, *result, *filter;

    checkCudaErrors(cudaMallocManaged(&_8bit_signal, sizeof(char) * INPUT_SIGNAL_SIZE * BATCH_SIZE, cudaMemAttachGlobal));
    checkCudaErrors(cudaMallocManaged(&tmp_result1, sizeof(cufftReal) * INPUT_SIGNAL_SIZE * BATCH_SIZE, cudaMemAttachGlobal));
    checkCudaErrors(cudaMallocManaged(&tmp_result2, sizeof(cufftComplex) * COMPLEX_SIGNAL_SIZE * BATCH_SIZE, cudaMemAttachGlobal));
    checkCudaErrors(cudaMallocManaged(&result, sizeof(cufftComplex) * COMPLEX_SIGNAL_SIZE * BATCH_SIZE, cudaMemAttachGlobal));
    checkCudaErrors(cudaMallocManaged(&filter, sizeof(cufftComplex) * COMPLEX_SIGNAL_SIZE, cudaMemAttachGlobal));
    
    initInputs(_8bit_signal, filter);

    //compute reference result for later verification
    printf("Computing reference solution\n");
    cufftComplex *reference = computeReference(_8bit_signal, filter);

    printf("Creating FFT plan\n");
    cufftHandle fftPlan;
    size_t workSize;
    
    checkCudaErrors(cufftCreate(&fftPlan));
    int signalSize = INPUT_SIGNAL_SIZE;
    checkCudaErrors(cufftMakePlanMany(fftPlan, 1, &signalSize, 0,0,0,0,0,0, CUFFT_R2C, BATCH_SIZE, &workSize));
    
    //create timers
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float elapsedTime;

    // Perform computation
    printf("Running %d iterations%s\n", ITERATIONS, USE_OPTIMIZED_TRANSPOSE ? " (using optimized transpose)" : "");
    checkCudaErrors(cudaEventRecord(start, 0));    

    /*
     * The actual computation
     */

    dim3 block(TILE_DIM, BLOCK_ROWS);
    dim3 grid((COMPLEX_SIGNAL_SIZE + block.x - 1)/block.x, (BATCH_SIZE + block.y - 1)/block.y);

    for(int i = 0; i < ITERATIONS; i++) {
        //Step 1
        ConvertInputR<<<32, 128>>>(_8bit_signal, tmp_result1);
        checkCudaErrors(cudaGetLastError());
        
        //Step 2
        checkCudaErrors(cufftExecR2C(fftPlan, tmp_result1, tmp_result2));
        
        //Step 3
        if(USE_OPTIMIZED_TRANSPOSE)
            ConvolveAndStoreTransposedC_Optimized<<<grid, block>>>(tmp_result2, result, filter);
        else
            ConvolveAndStoreTransposedC_Basic<<<grid, block>>>(tmp_result2, result, filter);
        
        checkCudaErrors(cudaGetLastError());
    }

    checkCudaErrors(cudaEventRecord(end, 0));
    checkCudaErrors(cudaDeviceSynchronize());
    checkCudaErrors(cudaEventSynchronize(end));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, end));
    printf("Time for the FFT: %fms\n", elapsedTime);
    
    //Verify result
    if(postprocess(reference, result, COMPLEX_SIGNAL_SIZE * BATCH_SIZE)) {
        printf("Verification successful.\n");
    } else {
        printf("!!! Verification Failed !!!\n");
    }

    //Cleanup
    checkCudaErrors(cufftDestroy(fftPlan));

    checkCudaErrors(cudaFree(_8bit_signal));
    checkCudaErrors(cudaFree(tmp_result1));
    checkCudaErrors(cudaFree(tmp_result2));
    checkCudaErrors(cudaFree(result));
    checkCudaErrors(cudaFree(filter));
    checkCudaErrors(cudaFree(reference));

    //clean up driver state
    cudaDeviceReset();

    printf("Done\n");
    
    return 0;
}