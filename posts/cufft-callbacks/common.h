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

 
#define INPUT_SIGNAL_SIZE 1024
#define BATCH_SIZE 1000
#define COMPLEX_SIGNAL_SIZE (INPUT_SIGNAL_SIZE/2 + 1)
#define ITERATIONS 100

////////////////////////////////////////////////////////////////////////////////
// CUDA error checking
////////////////////////////////////////////////////////////////////////////////
#define checkCudaErrors(val)           __checkCudaErrors__ ( (val), #val, __FILE__, __LINE__ )
 
template <typename T>
inline void __checkCudaErrors__(T code, const char *func, const char *file, int line) 
{
    if (code) {
        fprintf(stderr, "CUDA error at %s:%d code=%d \"%s\" \n",
                file, line, (unsigned int)code, func);
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }
}

////////////////////////////////////////////////////////////////////////////////
// Helper functions
////////////////////////////////////////////////////////////////////////////////
__device__ __host__ inline cufftComplex ComplexMul(cufftComplex a, cufftComplex b)
{
    cufftComplex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

void initInputs(char *dataIn, cufftComplex *filter) {
	srand(42);

    // Initalize the memory for the signal
    for (size_t i = 0; i < INPUT_SIGNAL_SIZE * BATCH_SIZE; ++i)
    {
        if(i % INPUT_SIGNAL_SIZE == 0) srand(42);
        float val = rand() / (float)RAND_MAX;
        dataIn[i] = (char)(127 * val);
    }

    // Initialize correction vector
    for(size_t i = 0; i < COMPLEX_SIGNAL_SIZE; i++) {
        srand(42);
        filter[i].x = rand() / (float)RAND_MAX;
        filter[i].y = 0.5f;
    }
}

////////////////////////////////////////////////////////////////////////////////
// Verification
////////////////////////////////////////////////////////////////////////////////
bool postprocess(const cufftComplex *ref, const cufftComplex *res, int size)
{
  bool passed = true;
  for (int i = 0; i < size; i++)
    if (res[i].x != ref[i].x || res[i].y != ref[i].y) {
      printf("%d: (%4.2f,%4.2f) != (%4.2f, %4.2f)\n", i, res[i].x, res[i].y, ref[i].x, ref[i].y);
      printf("%25s\n", "*** FAILED ***");
      passed = false;
      break;
    }
  return passed;
}

//CPU Versions of the custom kernels
void ConvertInputR_onCPU(
        const char * __restrict__ dataIn, 
        cufftReal * __restrict__ dataOut, 
        size_t size) 
{
	for(size_t i = 0; i < size; i++) {
		char element = ((char*)dataIn)[i];
        dataOut[i] = (cufftReal)((float)element/127.0f);
	}
}

void ConvolveAndStoreTransposedC_onCPU(
    const cufftComplex * __restrict__ dataIn, 
    cufftComplex * __restrict__ dataOut,
    const cufftComplex * __restrict__ filter)
{
	for(size_t row = 0; row < BATCH_SIZE; row++) {
		for(size_t col = 0; col < COMPLEX_SIGNAL_SIZE; col++) {
			cufftComplex value = ComplexMul(dataIn[row * COMPLEX_SIGNAL_SIZE + col], filter[col]);
			dataOut[col * BATCH_SIZE + row] = value;
		}
	}
}

/*
 * computes the reference and returns a newly allcoated cuda-managed cufftComplex* array.
 * The caller is responsible for freeing the reference array
  */
cufftComplex* computeReference(const char *dataIn, const cufftComplex *filter) {
    cufftReal *tmp_result1;
    cufftComplex *tmp_result2, *result;

    checkCudaErrors(cudaMallocManaged(&tmp_result1, sizeof(cufftReal) * INPUT_SIGNAL_SIZE * BATCH_SIZE, cudaMemAttachGlobal));
    checkCudaErrors(cudaMallocManaged(&tmp_result2, sizeof(cufftComplex) * COMPLEX_SIGNAL_SIZE * BATCH_SIZE, cudaMemAttachGlobal));
    checkCudaErrors(cudaMallocManaged(&result, sizeof(cufftComplex) * COMPLEX_SIGNAL_SIZE * BATCH_SIZE, cudaMemAttachGlobal));

    ConvertInputR_onCPU(dataIn, tmp_result1, INPUT_SIGNAL_SIZE * BATCH_SIZE);

    //We use cuFFT to compute the reference; we want to verify the callbacks and custom kernels
    //and not the FFT itself, so that's fine
    cufftHandle fftPlan;
    size_t workSize;
    
    checkCudaErrors(cufftCreate(&fftPlan));
    int signalSize = INPUT_SIGNAL_SIZE;
    checkCudaErrors(cufftMakePlanMany(fftPlan, 1, &signalSize, 0,0,0,0,0,0, CUFFT_R2C, BATCH_SIZE, &workSize));

    checkCudaErrors(cufftExecR2C(fftPlan, tmp_result1, tmp_result2));
    checkCudaErrors(cudaDeviceSynchronize());

    ConvolveAndStoreTransposedC_onCPU(tmp_result2, result, filter);

    checkCudaErrors(cufftDestroy(fftPlan));
    checkCudaErrors(cudaFree(tmp_result1));
    checkCudaErrors(cudaFree(tmp_result2));
    
    return result;
}