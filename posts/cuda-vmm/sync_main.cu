/* Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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
#include <thread>
#include <vector>
#include <iostream>
#include <atomic>
#include <cuda.h>

static inline void
checkRtError(cudaError_t res, const char *tok, const char *file, unsigned line)
{
    if (res != cudaSuccess) {
        std::cerr << file << ':' << line << ' ' << tok
                  << "failed (" << (unsigned)res << "): " << cudaGetErrorString(res) << std::endl;
        abort();
    }
}

#define CHECK_RT(x) checkRtError(x, #x, __FILE__, __LINE__);

static inline void
checkDrvError(CUresult res, const char *tok, const char *file, unsigned line)
{
    if (res != CUDA_SUCCESS) {
        const char *errStr = NULL;
        (void)cuGetErrorString(res, &errStr);
        std::cerr << file << ':' << line << ' ' << tok
                  << "failed (" << (unsigned)res << "): " << errStr << std::endl;
        abort();
    }
}

#define CHECK_DRV(x) checkDrvError(x, #x, __FILE__, __LINE__);

__global__ void spinKernel(unsigned long long timeout_clocks = 100000ULL)
{
    register unsigned long long start_time, sample_time;
    start_time = clock64();
    while(1) {
        sample_time = clock64();
        if (timeout_clocks != ~0ULL && (sample_time - start_time) > timeout_clocks) {
            break;
        }
    }
}

class MMAPAllocation {
    size_t sz;
    CUmemGenericAllocationHandle hdl;
    CUmemAccessDesc accessDesc;
    CUdeviceptr ptr;
public:
    MMAPAllocation(size_t size, int dev = 0) {
        size_t aligned_sz;
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = dev;
        accessDesc.location = prop.location;
        accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        CHECK_DRV(cuMemGetAllocationGranularity(&aligned_sz, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
        sz = ((size + aligned_sz - 1) / aligned_sz) * aligned_sz;

        CHECK_DRV(cuMemAddressReserve(&ptr, sz, 0ULL, 0ULL, 0ULL));
        CHECK_DRV(cuMemCreate(&hdl, sz, &prop, 0));
        CHECK_DRV(cuMemMap(ptr, sz, 0ULL, hdl, 0ULL));
        CHECK_DRV(cuMemSetAccess(ptr, sz, &accessDesc, 1ULL));
    }
    ~MMAPAllocation() {
        CHECK_DRV(cuMemUnmap(ptr, sz));
        CHECK_DRV(cuMemAddressFree(ptr, sz));
        CHECK_DRV(cuMemRelease(hdl));
    }
};

void launch_work(std::atomic<bool> &keep_going, std::atomic<unsigned> &ready, cudaStream_t stream)
{
    spinKernel<<<1,1,0,stream>>>();
    CHECK_RT(cudaGetLastError());

    // We've launched at least one thing, tell the master thread
    ready.fetch_add(1, std::memory_order_release);

    while(keep_going.load(std::memory_order_acquire)) {
        spinKernel<<<1,1,0,stream>>>();
        CHECK_RT(cudaGetLastError());
    }
}


int main()
{
    const size_t N = 4ULL;
    std::atomic<bool> keep_going(true);
    std::atomic<unsigned> ready(0);
    std::vector<std::thread> threads;
    std::vector<cudaStream_t> streams;
    int supportsVMM = 0;
    CUdevice dev;

    CHECK_RT(cudaFree(0));  // Force and check the initialization of the runtime

    CHECK_DRV(cuCtxGetDevice(&dev));
    CHECK_DRV(cuDeviceGetAttribute(&supportsVMM, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, dev));

    for (size_t i = 0; i < N; i++) {
        cudaStream_t stream;
        CHECK_RT(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        std::thread t1(launch_work, std::ref(keep_going), std::ref(ready), stream);

        threads.push_back(std::move(t1));
        streams.push_back(stream);
    }

    // Wait for all the threads to have launched at least one thing
    while (ready.load(std::memory_order_acquire) != N);

    // Use standard cudaMalloc/cudaFree
    for (size_t i = 0; i < 100; i++) {
        int *x = nullptr;
        CHECK_RT(cudaMalloc(&x, sizeof(*x)));
        CHECK_RT(cudaFree(x));
    }


    if (supportsVMM) {
        // Now use the Virtual Memory Management APIs
        for (size_t i = 0; i < 100; i++) {
            MMAPAllocation allocMMAP(1);
        }
    }

    keep_going.store(false, std::memory_order_release);

    for (size_t i = 0; i <threads.size(); i++) {
        threads[i].join();
        CHECK_RT(cudaStreamDestroy(streams[i]));
    }

    return 0;
}
