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
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <assert.h>

#include "cuvector.h"

typedef std::chrono::nanoseconds ReportingDuration;

static inline void
checkDrvError(CUresult res, const char *tok, const char *file, unsigned line)
{
    if (res != CUDA_SUCCESS) {
        const char *errStr = NULL;
        (void)cuGetErrorString(res, &errStr);
        std::cerr << file << ':' << line << ' ' << tok
                  << "failed (" << (unsigned)res << "): " << errStr << std::endl;
    }
}

#define CHECK_DRV(x) checkDrvError(x, #x, __FILE__, __LINE__);

template<typename V>
void measureGrow(V& v, size_t minN, size_t maxN, std::vector<ReportingDuration>& durations)
{
    for (size_t n = minN; n <= maxN; n <<= 1) {
        typedef std::chrono::time_point<std::chrono::steady_clock> time_point;

        time_point start = std::chrono::steady_clock::now();
        CUresult status = v.grow(n);
        time_point end = std::chrono::steady_clock::now();

        durations.push_back(std::chrono::duration_cast<ReportingDuration>(end - start));
        // In non-release, verify the memory is accessible and everything worked properly
        assert(CUDA_SUCCESS == status);
        assert(CUDA_SUCCESS == cuMemsetD8((CUdeviceptr)v.getPointer(), 0, v.getSize()));
        assert(CUDA_SUCCESS == cuCtxSynchronize());
    }
}

template<typename Allocator, typename Elem>
void runVectorPerfTest(CUcontext ctx, size_t minN, size_t maxN,
                       std::vector<ReportingDuration>& noReserveDurations,
                       std::vector<ReportingDuration>& reserveDurations)
{
    typedef cuda_utils::Vector<Elem, Allocator> VectorDUT;

    if (false) {
        // Warm-up
        VectorDUT dut(ctx);
        if (!dut.grow(maxN)) {
            std::cerr << "Failed to grow to max elements, test invalid!\n" << std::endl;
            return;
        }
    }

    // Wait for the OS to settle it's GPU pages from past perf runs
    std::this_thread::sleep_for(std::chrono::seconds(2));
    {
        // Measure without reserving
        VectorDUT dut(ctx);
        measureGrow(dut, minN, maxN, noReserveDurations);
    }

    // Wait for the OS to settle it's GPU pages from past perf runs
    std::this_thread::sleep_for(std::chrono::seconds(2));
    {
        size_t free = 0ULL;
        VectorDUT dut(ctx);

        dut.reserve(maxN);
        CHECK_DRV(cuMemGetInfo(&free, NULL));
        std::cout << "\tReserved " << maxN << " elements..." << std::endl
                  << "\tFree Memory: " << (float)free / std::giga::num << "GB" << std::endl;

        measureGrow(dut, minN, maxN, reserveDurations);
    }
}

int main()
{
    size_t free;
    typedef unsigned char ElemType;
    CUcontext ctx;
    CUdevice dev;
    int supportsVMM = 0;

    CHECK_DRV(cuInit(0));
    CHECK_DRV(cuDevicePrimaryCtxRetain(&ctx, 0));
    CHECK_DRV(cuCtxSetCurrent(ctx));
    CHECK_DRV(cuCtxGetDevice(&dev));

    std::vector<std::vector<ReportingDuration> > durations(4);

    CHECK_DRV(cuMemGetInfo(&free, NULL));

    std::cout << "Total Free Memory: " << (float)free / std::giga::num << "GB" << std::endl;

    // Skip the smaller cases
    const size_t minN = (2ULL * 1024ULL * 1024ULL + sizeof(ElemType) - 1ULL) / sizeof(ElemType);
    // Use at max about 75% of all vidmem for perf testing
    // Also, some vector allocators like MemAlloc cannot handle more than this,
    // as they would run out of memory during the grow algorithm
    const size_t maxN = 3ULL * free / (4ULL * sizeof(ElemType));

    std::cout << "====== cuMemAlloc ElemSz=" << sizeof(ElemType) << " ======" << std::endl;
    runVectorPerfTest<cuda_utils::VectorMemAlloc, ElemType>(ctx, minN, maxN, durations[0], durations[1]);
    std::cout << "====== cuMemAllocManaged ElemSz=" << sizeof(ElemType) << " ======" << std::endl;
    runVectorPerfTest<cuda_utils::VectorMemAllocManaged, ElemType>(ctx, minN, maxN, durations[2], durations[3]);

    CHECK_DRV(cuDeviceGetAttribute(&supportsVMM, CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED, dev));

    if (supportsVMM) {
        durations.resize(durations.size() + 2);
        std::cout << "====== cuMemMap ElemSz=" << sizeof(ElemType) << " ======" << std::endl;
        runVectorPerfTest<cuda_utils::VectorMemMap, ElemType>(ctx, minN, maxN, durations[4], durations[5]);
    }

    // Quick and dirty table of results
    std::cout << "Size(bytes)    | "
              << "Alloc(us)      | "
              << "AllocRes(us)   | "
              << "Managed(us)    | "
              << "ManagedRes(us) | ";

    if (supportsVMM) {
        std::cout << "cuMemMap(us)   | "
                  << "cuMemMapRes(us)| ";
    }

    std::cout << std::endl;

    for (size_t i = 0; i < durations[0].size(); i++) {
        std::cout << std::left << std::setw(15) << std::setfill(' ') << (minN << i) << "| ";
        for (size_t j = 0; j < durations.size(); j++) {
            std::cout << std::left << std::setw(15) << std::setfill(' ')
                      << std::setprecision(2) << std::fixed
                      << std::chrono::duration_cast <std::chrono::duration<float, std::micro> >(durations[j][i]).count() << "| ";
        }
        std::cout << std::endl;
    }

    CHECK_DRV(cuDevicePrimaryCtxRelease(0));

    return 0;
}
