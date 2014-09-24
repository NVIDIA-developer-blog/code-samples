/*
 * Copyright 1993-2014 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
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