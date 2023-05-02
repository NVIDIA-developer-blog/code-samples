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

#pragma once
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include "cuda.h"
#include <memory>
#include <cuda/atomic>

namespace CASError {

  enum AtomicStatus {
    ATOMIC_NO_ERROR = 0,
    ATOMIC_ERROR_REPORTED = 1
  };

  inline
  cudaError_t checkCuda(cudaError_t result)
  {
  #if defined(DEBUG) || defined(_DEBUG)
    if (result != cudaSuccess) {
      fprintf(stderr, "CUDA Runtime Error: %s\n", 
              cudaGetErrorString(result));
      assert(result == cudaSuccess);
    }
  #endif
    return result;
  }

  // Allocates system-pinned memory of type ErrorType
  template < typename ErrorType>
  struct PinnedMemory 
  {
      using data_type = ErrorType;

      PinnedMemory() {
          checkCuda(cudaMallocHost(&hdata, sizeof(data_type)));
      }

      ~PinnedMemory(){
          cudaFreeHost(hdata);
      }
      data_type *hdata;
  };

  // DeviceStatus allocates system-pinned memory of StatusType and also allocates corresponding device memory of StatusType
  template <typename StatusType>
  struct DeviceStatus
  {

      using status_type = StatusType;

      DeviceStatus () {
          checkCuda(cudaMallocHost(&host_status, sizeof(status_type), cudaHostAllocMapped));
          checkCuda(cudaMalloc(&device_status, sizeof(status_type)));
      }

      ~DeviceStatus() {
          checkCuda(cudaFreeHost(host_status));
          checkCuda(cudaFree(device_status));
      }    

      status_type __host__ status() {
        return static_cast<volatile cuda::std::atomic<StatusType> *>(host_status)->load(cuda::memory_order_acquire);
      }

      cuda::std::atomic<StatusType> *host_status;
      StatusType *device_status;
  };

  // This struct represents the data accessible and modifiable on the device and contains pointers to relevant information
  template <typename ErrorType>
  struct MappedErrorTypeDeviceData {
        using status_type = AtomicStatus;
        // these two members are used so that they can be accessed from the device directly
        cuda::std::atomic<status_type> * host_status;
        status_type * device_status;

        // pointer to pinned data to be accessed from the device directly
        ErrorType * host_data; 

        void inline __device__ synchronizeStatus() {
          host_status->store(ATOMIC_ERROR_REPORTED, cuda::memory_order_release);
        }   
      };


  /* The MappedErrorType creates system-pinned memory of ErrorType, as well as corresponding DeviceStatus. 
  * Using the available methods guarantees the necessary memory fences to avoid asynchronous race conditions.
  */ 

  template <typename ErrorType>
  struct MappedErrorType 
  {
      // Use same status type as MappedErrorTypeDeviceData
      using status_type = typename MappedErrorTypeDeviceData<ErrorType>::status_type;

      // System-pinned error payload to be written by supplied function
      std::shared_ptr<PinnedMemory<ErrorType>> error_data;

      // Error reporting indicator to coordinate asynchronous soft error reporting
      std::shared_ptr<DeviceStatus<status_type>> status;

      // The necessary device-side pointers needed for proper reporting
      MappedErrorTypeDeviceData<ErrorType> deviceData;

      MappedErrorType (cudaStream_t stream = 0) 
      : error_data(new PinnedMemory<ErrorType>()),
        status(new DeviceStatus<status_type>()),
        deviceData (MappedErrorTypeDeviceData<ErrorType>{.host_status =status->host_status,
                              .device_status =status->device_status, 
                              .host_data = error_data->hdata})
      { 
        deviceData.host_status->store(ATOMIC_NO_ERROR, cuda::memory_order_release);
        checkCuda(cudaMemsetAsync(deviceData.device_status, ATOMIC_NO_ERROR, sizeof(status_type), stream));
      }

      /// Checks on the host if an error has been reported
      bool __host__ checkErrorReported() {
        return (status->status() == ATOMIC_ERROR_REPORTED);
      }

      /** Returns host-pinned error payload
      * Note: If error data includes device pointers (e.g. const char*) you will need to properly post-processes these pointers
      */
      volatile ErrorType & __host__ get() {
        return *static_cast<volatile ErrorType *>(deviceData.host_data);
      }

      /// Clears both the device-side status and host-side 
      void __host__ clear(cudaStream_t stream = 0) {
        checkCuda(cudaMemsetAsync(deviceData.device_status, ATOMIC_NO_ERROR, sizeof(status_type), stream));
        (deviceData.host_status)->store(ATOMIC_NO_ERROR, cuda::memory_order_release);
      }

      void inline __device__ synchronizeStatus() {
        deviceData.synchronizeStatus();
      }

  };

  /// Retrieve variable like __FILE__ when set in device memory to host
  std::string getDeviceString(const char * device_string, cudaStream_t stream = 0) {
      CUdeviceptr pbase;
      std::size_t psize;
      cuMemGetAddressRange(&pbase, &psize, reinterpret_cast<CUdeviceptr>(device_string));
      std::string str;
      str.resize(psize);
      cudaMemcpyAsync(str.data(), device_string, psize, cudaMemcpyDeviceToHost, stream);    
      return str;  
  }

  template <typename ErrorType, typename FunctionType>
  inline __device__ void report_first_error(MappedErrorType<ErrorType> & error_dat, FunctionType func){
      if (atomicCAS(reinterpret_cast<int*>(error_dat.deviceData.device_status), static_cast<int>(ATOMIC_NO_ERROR), static_cast<int>(ATOMIC_ERROR_REPORTED)) == static_cast<int>(ATOMIC_NO_ERROR) ) {
          func(*error_dat.deviceData.host_data);
          __threadfence_system();
          error_dat.synchronizeStatus();
      }
  }

}