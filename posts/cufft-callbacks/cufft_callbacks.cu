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

////////////////////////////////////////////////////////////////////////////////
// Callback Implementations
////////////////////////////////////////////////////////////////////////////////
__device__ cufftReal CB_ConvertInputR(void *dataIn, size_t offset, void *callerInfo, void *sharedPtr) {
    char element = ((char*)dataIn)[offset];
    return (cufftReal)((float)element/127.0f);
}

__device__ cufftCallbackLoadR d_loadCallbackPtr = CB_ConvertInputR; 

__device__ void CB_ConvolveAndStoreTransposedC(void *dataOut, size_t offset, cufftComplex element, void *callerInfo, void *sharedPtr) {
    cufftComplex *filter = (cufftComplex*)callerInfo;
    size_t row = offset / COMPLEX_SIGNAL_SIZE;
    size_t col = offset % COMPLEX_SIGNAL_SIZE;

    ((cufftComplex*)dataOut)[col * BATCH_SIZE + row] = ComplexMul(element, filter[col]);
}

__device__ cufftCallbackStoreC d_storeCallbackPtr = CB_ConvolveAndStoreTransposedC;

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
    cufftComplex *result, *filter;

    checkCudaErrors(cudaMallocManaged(&_8bit_signal, sizeof(char) * INPUT_SIGNAL_SIZE * BATCH_SIZE, cudaMemAttachGlobal));
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

    /*
     * Retrieve address of callback functions on the device
     */                              
    cufftCallbackLoadR h_loadCallbackPtr;
    cufftCallbackStoreC h_storeCallbackPtr;
    checkCudaErrors(cudaMemcpyFromSymbol(&h_loadCallbackPtr, 
                                          d_loadCallbackPtr, 
                                          sizeof(h_loadCallbackPtr)));
    checkCudaErrors(cudaMemcpyFromSymbol(&h_storeCallbackPtr, 
                                          d_storeCallbackPtr, 
                                          sizeof(h_storeCallbackPtr)));

    // Now associate the callbacks with the plan.
    cufftResult status = cufftXtSetCallback(fftPlan, 
                            (void **)&h_loadCallbackPtr, 
                            CUFFT_CB_LD_REAL,
                            0);
    if (status == CUFFT_LICENSE_ERROR) {
        printf("This sample requires a valid license file.\n");
        printf("The file was either not found, out of date, or otherwise invalid.\n");
        exit(EXIT_FAILURE);
    } else {
        checkCudaErrors(status);
    }

   checkCudaErrors(cufftXtSetCallback(fftPlan, 
                                (void **)&h_storeCallbackPtr, 
                                CUFFT_CB_ST_COMPLEX,
                                (void **)&filter));

    //create timers
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    float elapsedTime;

    printf("Running %d iterations\n", ITERATIONS);
    checkCudaErrors(cudaEventRecord(start, 0));

    /*
     * The actual Computation
     */

    for(int i = 0; i < ITERATIONS; i++) {
        checkCudaErrors(cufftExecR2C(fftPlan, (cufftReal*)_8bit_signal, result));
    }

    checkCudaErrors(cudaEventRecord(end, 0));
    checkCudaErrors(cudaEventSynchronize(end));
    checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, end));
    printf("Time for the FFT: %fms\n", elapsedTime);

    //Verify correct result    
    if(postprocess(reference, result, COMPLEX_SIGNAL_SIZE * BATCH_SIZE)) {
        printf("Verification successful.\n");
    } else {
        printf("!!! Verification Failed !!!\n");
    }

    //Cleanup
    checkCudaErrors(cufftDestroy(fftPlan));

    checkCudaErrors(cudaFree(_8bit_signal));
    checkCudaErrors(cudaFree(result));
    checkCudaErrors(cudaFree(filter));
    checkCudaErrors(cudaFree(reference));
      
    //clean up driver state
    cudaDeviceReset();

    printf("Done\n");
    
    return 0;
}