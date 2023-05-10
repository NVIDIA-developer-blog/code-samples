/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <iostream>
#include <stdio.h>
#include <assert.h>
#include "cuda.h"
#include "CASerror.h"

enum codes
{
    NO_ERROR = 0,
    RANGE_ERROR = 1,
    LARGE_VALUE_ERROR = 2,
    NEG_SQRT_ERROR = 3,
    UNSPECIFIED_ERROR = 999,
};

struct RandomSpikeError {
    
    int code;
    int line;
    int filenum;
    int block;
    int thread;
    // payload information
    int idx;
    float val;
};

struct OtherError {
    
    int code;
    int line;    
    int block;
    int thread;
    const char* file;
    int idx;
};

__global__ void randomSpikeKernel(float* out, int sz)
// This kernel generates a pseudo-random number 
// then puts it into 1/num-100+1e-6. That curve will be 
// sharply peaked at num=100 where the value will be 1e6.
// In the case of a very large value, we want to report an
// error without stopping the kernel.
{
   for (int idx = threadIdx.x + blockIdx.x * blockDim.x;
       idx < sz;
       idx += blockDim.x * gridDim.x)
   {
       const int A = 187;
       const int M = 7211;
       int ival = ((idx + A) * A) % M;
       ival = (ival*A) % M;
       ival = (ival*A) % M;
       float val = 1.f/(ival-100+1e-6);
       
       out[idx] = val;
   } 
}

__global__ void randomSpikeKernelwAssert(float* out, int sz)
// This kernel generates a pseudo-random number 
// then puts it into 1/num-100+1e-6. That curve will be 
// sharply peaked at num=100 where the value will be 1e6.
// In the case of a very large value, we want to report an
// error without stopping the kernel.
{
   for (int idx = threadIdx.x + blockIdx.x * blockDim.x;
       idx < sz;
       idx += blockDim.x * gridDim.x)
   {
       const int A = 187;
       const int M = 7211;
       int ival = ((idx + A) * A) % M;
       ival = (ival*A) % M;
       ival = (ival*A) % M;
       float val = 1.f/(ival-100+1e-6);
       assert(val < 10000);
       
       out[idx] = val;
   } 
}
__global__ void randomSpikeKernelwError(float* out, int sz)
// This kernel generates a pseudo-random number 
// then puts it into 1/num-100+1e-6. That curve will be 
// sharply peaked at num=100 where the value will be 1e6.
// In the case of a very large value, we want to report an
// error without stopping the kernel.
{
   for (int idx = threadIdx.x + blockIdx.x * blockDim.x;
       idx < sz;
       idx += blockDim.x * gridDim.x)
   {
       const int A = 187;
       const int M = 7211;
       int ival = ((idx + A) * A) % M;
       ival = (ival*A) % M;
       ival = (ival*A) % M;
       float val = 1.f/(ival-100+1e-6);
       
       if (val >= 10000) {
//          assert(val < sz);
            printf("val (%f) out of range for idx = %d\n", val, idx);
       }
       out[idx] = val;
   } 
}

__global__ void randomSpikeKernelFinal(float* out, int sz, CASError::MappedErrorType<RandomSpikeError> device_error_data)
// This kernel generates a pseudo-random number 
// then puts it into 1/num-100+1e-6. That curve will be 
// sharply peaked at num=100 where the value will be 1e6.
// In the case of a very large value, we want to report an
// error without stopping the kernel.
{
   for (int idx = threadIdx.x + blockIdx.x * blockDim.x;
       idx < sz;
       idx += blockDim.x * gridDim.x)
   {
       const int A = 187;
       const int M = 7211;
       int ival = ((idx + A) * A) % M;
       ival = (ival*A) % M;
       ival = (ival*A) % M;
       float val = 1.f/(ival-100+1e-6);
       
       if (val >= 10000) {
        report_first_error(device_error_data, [&] (auto &error){
               error = RandomSpikeError {
                  .code = LARGE_VALUE_ERROR,
                  .line = __LINE__,
                  .filenum = 0,
                  .block = static_cast<int>(blockIdx.x),
                  .thread = static_cast<int>(threadIdx.x),
                  .idx = idx,
                  .val = val
               };
        });        
       }
       out[idx] = val;
   } 
}

__global__ void otherKernel(float* inout, int sz, CASError::MappedErrorType<OtherError> device_error_data)
// This kernel computes the sqrt of the input values. Where the input
// value is negative, we raise an error
{
   for (int idx = threadIdx.x + blockIdx.x * blockDim.x;
       idx < sz;
       idx += blockDim.x * gridDim.x)
   {
      
       float val = inout[idx];
       if (val < 0) {
         report_first_error(device_error_data, [&] (auto &error){
               error = OtherError {
                  .code = NEG_SQRT_ERROR,
                  .line = __LINE__,
                  .block = static_cast<int>(blockIdx.x),               
                  .thread = static_cast<int>(threadIdx.x),
                  .file = __FILE__,
                  .idx = idx
               };
        });        
       } else inout[idx] = sqrt(val);
   }
}

int reportError( CASError::MappedErrorType<RandomSpikeError> & error_dat)
{
   int retval = NO_ERROR;
   
   if (error_dat.checkErrorReported()) {
      auto & error = error_dat.get();
      retval = error.code;
      std::cerr << "ERROR " << error.code
                << ", line " << error.line
                << ". block " << error.block
                << ", thread " << error.thread;
      if (retval == LARGE_VALUE_ERROR)
        std::cerr << ", value = " << error.val;
      std::cerr << std::endl;     
   }
   
   return retval;
}

int reportError( CASError::MappedErrorType<OtherError> & error_dat, cudaStream_t stream = 0)
{
   int retval = NO_ERROR;
   
   if (error_dat.checkErrorReported()) {
      auto & error = error_dat.get();
      retval = error.code;
      std::cerr << "ERROR " << error.code
                << ", line " << error.line
                << ", file " << CASError::getDeviceString(error.file, stream)
                << ". block " << error.block
                << ", thread " << error.thread;
      std::cerr << std::endl;     
   }
   
   return retval;
}

#define MAX_IDX 7000
int main(void)
{
   int device;
   cudaDeviceProp prop;
   CASError::checkCuda(cudaGetDevice(&device));
   CASError::checkCuda(cudaGetDeviceProperties(&prop, device));

   if (prop.concurrentManagedAccess) std::cout << "concurrentManagedAccess supported" << std::endl;
   if (prop.hostNativeAtomicSupported) std::cout << "hostNativeAtomicSupported supported" << std::endl;

   // Create pinned flags/data and device-side atomic flag for CAS
   auto mapped_error = CASError::MappedErrorType<RandomSpikeError>();
   auto mapped_error2 = CASError::MappedErrorType<OtherError>();

   int async_err;

   // Streams and events
   cudaStream_t stream; cudaStreamCreate(&stream);
   cudaEvent_t finishedRandomSpikeKernel;
   CASError::checkCuda( cudaEventCreate(&finishedRandomSpikeKernel) );
   cudaEvent_t finishedOtherKernel;
   CASError::checkCuda( cudaEventCreate(&finishedOtherKernel) );   

   float *out;
   auto h_out = new float [MAX_IDX];
   cudaMalloc((void**)&out, sizeof(float)*MAX_IDX);

   randomSpikeKernel<<<100,32,0,stream>>>(out, MAX_IDX);

   randomSpikeKernelFinal<<<100,32,0,stream>>>(out, MAX_IDX, mapped_error);
   CASError::checkCuda( cudaEventRecord(finishedRandomSpikeKernel, stream) );
   
#if 0
   CASError::checkCuda(cudaEventSynchronize(finishedRandomSpikeKernel));
#endif
   // Check the error message from err_data
   async_err = reportError(mapped_error);
   if (async_err != NO_ERROR) std::cout << "ERROR! " << "code: " << async_err << std::endl;
   else std::cout << "No error" << std::endl;
   otherKernel<<<100,32,0,stream>>>(out, MAX_IDX,  mapped_error2);
   CASError::checkCuda( cudaEventRecord(finishedOtherKernel, stream) );

#if 0
   CASError::checkCuda(cudaEventSynchronize(finishedOtherKernel));
#endif
   async_err = reportError(mapped_error2, stream);
   if (async_err != NO_ERROR) std::cout << "ERROR! " << "code: " << async_err << std::endl;
   else std::cout << "No error" << std::endl;

   std::cout << "Launch memcpy" << std::endl;
   cudaMemcpyAsync(h_out, out, sizeof(float)*MAX_IDX, cudaMemcpyDeviceToHost, stream);
   cudaStreamSynchronize(stream);
   async_err = reportError(mapped_error);   
   if (async_err != NO_ERROR) std::cout << "ERROR! " << "code: " << async_err << std::endl;
   else std::cout << "No error" << std::endl;
   mapped_error.clear(stream);   

   async_err = reportError(mapped_error2, stream);
   if (async_err != NO_ERROR) std::cout << "ERROR! " << "code: " << async_err << std::endl;
   else std::cout << "No error" << std::endl;

   int final_err = reportError(mapped_error);
   if (final_err != NO_ERROR) std::cout << "ERROR! " << "code: " << final_err << std::endl;
   else std::cout << "No error" << std::endl;

   cudaFree(out);
   cudaStreamDestroy(stream);
   return 0;
}
