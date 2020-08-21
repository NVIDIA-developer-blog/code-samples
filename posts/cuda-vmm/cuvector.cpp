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
#include <cuda.h>
#include <assert.h>
#include "cuvector.h"

// **************
// VectorMemAlloc
// **************

namespace cuda_utils {

VectorMemAlloc::VectorMemAlloc(CUcontext context) : ctx(context), d_p(0ULL), alloc_sz(0ULL)
{

}

VectorMemAlloc::~VectorMemAlloc()
{
    (void)cuMemFree(d_p);
}

// Although we're not supposed to "commit" memory in a reserve call,
// doing so for this sample demonstrates why reserve is so important
CUresult
VectorMemAlloc::reserve(size_t new_sz)
{
    CUresult status = CUDA_SUCCESS;
    CUdeviceptr new_ptr = 0ULL;
    CUcontext prev_ctx;

    if (new_sz <= alloc_sz) {
        return CUDA_SUCCESS;
    }
    (void)cuCtxGetCurrent(&prev_ctx);
    // Make sure we allocate on the correct context
    if ((status = cuCtxSetCurrent(ctx)) != CUDA_SUCCESS) {
        return status;
    }
    // Allocate the bigger buffer
    if ((status = cuMemAlloc(&new_ptr, new_sz)) == CUDA_SUCCESS) {
        // Copy over the bigger buffer.  We'll explicitly use the per thread
        // stream to ensure we don't add false dependencies on other threads
        // using the null stream, but we may have issues with other prior
        // work on this stream.  Luckily, that's not the case in our sample.
        //
        // We only want to copy over the alloc_sz here, as that's what's
        // actually committed at the moment
        if ((status = cuMemcpyAsync(new_ptr, d_p, alloc_sz, CU_STREAM_PER_THREAD)) == CUDA_SUCCESS) {
            // Free the smaller buffer.  We don't need to synchronize
            // CU_STREAM_PER_THREAD, since cuMemFree synchronizes for us
            (void)cuMemFree(d_p);
            d_p = new_ptr;
            alloc_sz = new_sz;
        }
        else {
            // Failed to copy the bigger buffer, free the smaller one
            (void)cuMemFree(new_ptr);
        }
    }
    // Make sure to always return to the previous context the caller had
    (void)cuCtxSetCurrent(prev_ctx);

    return status;
}

// *********************
// VectorMemAllocManaged
// *********************

VectorMemAllocManaged::VectorMemAllocManaged(CUcontext context) : ctx(context), dev(CU_DEVICE_INVALID), d_p(0ULL),
    alloc_sz(0ULL), reserve_sz(0ULL)
{
    CUcontext prev_ctx;
    (void)cuCtxGetCurrent(&prev_ctx);
    if (cuCtxSetCurrent(context) == CUDA_SUCCESS) {
        (void)cuCtxGetDevice(&dev);
        (void)cuCtxSetCurrent(prev_ctx);
    }

    (void)cuDeviceGetAttribute(&supportsConcurrentManagedAccess, CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, dev);
}

VectorMemAllocManaged::~VectorMemAllocManaged()
{
    (void)cuMemFree(d_p);
}

CUresult
VectorMemAllocManaged::reserve(size_t new_sz)
{
    CUresult status = CUDA_SUCCESS;
    CUcontext prev_ctx;
    CUdeviceptr new_ptr = 0ULL;

    if (new_sz <= reserve_sz) {
        return CUDA_SUCCESS;
    }

    (void)cuCtxGetCurrent(&prev_ctx);
    if ((status = cuCtxSetCurrent(ctx)) != CUDA_SUCCESS) {
        return status;
    }

    // Allocate the bigger buffer
    if ((status = cuMemAllocManaged(&new_ptr, new_sz, CU_MEM_ATTACH_GLOBAL)) == CUDA_SUCCESS) {
        // Set the preferred location for this managed allocation, to bias
        // any migration requests ("pinning" it under most circumstances to
        // the requested device)
        (void)cuMemAdvise(new_ptr, new_sz, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, dev);
        // Copy over the bigger buffer.  We'll explicitly use the per thread
        // stream to ensure we don't add false dependencies on other threads
        // using the null stream, but we may have issues with other prior
        // work on this stream.  Luckily, that's not the case in our sample.
        //
        // We only want to copy over the alloc_sz here, as that's what's
        // actually committed at the moment
        if (alloc_sz > 0) {
            if ((status = cuMemcpyAsync(new_ptr, d_p, alloc_sz, CU_STREAM_PER_THREAD)) == CUDA_SUCCESS) {
                // Free the smaller buffer.  We don't need to synchronize
                // CU_STREAM_PER_THREAD, since cuMemFree synchronizes for us
                (void)cuMemFree(d_p);
            }
            else {
                // Failed to copy the bigger buffer, free the smaller one
                (void)cuMemFree(new_ptr);
            }
        }
        if (status == CUDA_SUCCESS) {
            d_p = new_ptr;
            reserve_sz = new_sz;
        }
    }

    // Make sure to always return to the previous context the caller had
    (void)cuCtxSetCurrent(prev_ctx);

    return status;
}

// Actually commits num bytes of additional memory
CUresult
VectorMemAllocManaged::grow(size_t new_sz)
{
    CUresult status = CUDA_SUCCESS;
    CUcontext prev_ctx;

    if (new_sz <= alloc_sz) {
        return CUDA_SUCCESS;
    }
    if ((status = reserve(new_sz)) != CUDA_SUCCESS) {
        return status;
    }

    (void)cuCtxGetCurrent(&prev_ctx);
    // Make sure we allocate on the correct context
    if ((status = cuCtxSetCurrent(ctx)) != CUDA_SUCCESS) {
        return status;
    }
    // Actually commit the needed memory
    // We explicitly use the per thread stream here to ensure we're not
    // conflicting with other uses of the null stream from other threads
    if (supportsConcurrentManagedAccess &&
        (status = cuMemPrefetchAsync(d_p + alloc_sz, (new_sz - alloc_sz), dev,
                                     CU_STREAM_PER_THREAD)) == CUDA_SUCCESS) {
        // Not completely necessary, but will ensure the prefetch is complete
        // and prevent future runtime faults.  Also makes for a more fair
        // benchmark comparision
        if ((status = cuStreamSynchronize(CU_STREAM_PER_THREAD)) == CUDA_SUCCESS) {
            alloc_sz = new_sz;
        }
    }
    // Make sure to always return to the previous context the caller had
    (void)cuCtxSetCurrent(prev_ctx);
    return status;
}

// *********************
// VectorMemMap
// *********************

VectorMemMap::VectorMemMap(CUcontext context) : d_p(0ULL), prop(), handles(), alloc_sz(0ULL), reserve_sz(0ULL), chunk_sz(0ULL)
{
    CUdevice device;
    CUcontext prev_ctx;
    CUresult status = CUDA_SUCCESS;
    (void)status;

    status = cuCtxGetCurrent(&prev_ctx);
    assert(status == CUDA_SUCCESS);
    if (cuCtxSetCurrent(context) == CUDA_SUCCESS) {
        status = cuCtxGetDevice(&device);
        assert(status == CUDA_SUCCESS);
        status = cuCtxSetCurrent(prev_ctx);
        assert(status == CUDA_SUCCESS);
    }

    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = (int)device;
    prop.win32HandleMetaData = NULL;

    accessDesc.location = prop.location;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    status = cuMemGetAllocationGranularity(&chunk_sz, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
    assert(status == CUDA_SUCCESS);
}

VectorMemMap::~VectorMemMap()
{
    CUresult status = CUDA_SUCCESS;
    (void)status;
    if (d_p != 0ULL) {
        status = cuMemUnmap(d_p, alloc_sz);
        assert(status == CUDA_SUCCESS);
        for (size_t i = 0ULL; i < va_ranges.size(); i++) {
            status = cuMemAddressFree(va_ranges[i].start, va_ranges[i].sz);
            assert(status == CUDA_SUCCESS);
        }
        for (size_t i = 0ULL; i < handles.size(); i++) {
            status = cuMemRelease(handles[i]);
            assert(status == CUDA_SUCCESS);
        }
    }
}

CUresult
VectorMemMap::reserve(size_t new_sz)
{
    CUresult status = CUDA_SUCCESS;
    CUdeviceptr new_ptr = 0ULL;

    if (new_sz <= reserve_sz) {
        return CUDA_SUCCESS;
    }

    const size_t aligned_sz = ((new_sz + chunk_sz - 1) / chunk_sz) * chunk_sz;

    status = cuMemAddressReserve(&new_ptr, (aligned_sz - reserve_sz), 0ULL, d_p + reserve_sz, 0ULL);

    // Try to reserve an address just after what we already have reserved
    if (status != CUDA_SUCCESS || (new_ptr != d_p + reserve_sz)) {
        if (new_ptr != 0ULL) {
            (void)cuMemAddressFree(new_ptr, (aligned_sz - reserve_sz));
        }
        // Slow path - try to find a new address reservation big enough for us
        status = cuMemAddressReserve(&new_ptr, aligned_sz, 0ULL, 0U, 0);
        if (status == CUDA_SUCCESS && d_p != 0ULL) {
            CUdeviceptr ptr = new_ptr;
            // Found one, now unmap our previous allocations
            status = cuMemUnmap(d_p, alloc_sz);
            assert(status == CUDA_SUCCESS);
            for (size_t i = 0ULL; i < handles.size(); i++) {
                const size_t hdl_sz = handle_sizes[i];
                // And remap them, enabling their access
                if ((status = cuMemMap(ptr, hdl_sz, 0ULL, handles[i], 0ULL)) != CUDA_SUCCESS)
                    break;
                if ((status = cuMemSetAccess(ptr, hdl_sz, &accessDesc, 1ULL)) != CUDA_SUCCESS)
                    break;
                ptr += hdl_sz;
            }
            if (status != CUDA_SUCCESS) {
                // Failed the mapping somehow... clean up!
                status = cuMemUnmap(new_ptr, aligned_sz);
                assert(status == CUDA_SUCCESS);
                status = cuMemAddressFree(new_ptr, aligned_sz);
                assert(status == CUDA_SUCCESS);
            }
            else {
                // Clean up our old VA reservations!
                for (size_t i = 0ULL; i < va_ranges.size(); i++) {
                    (void)cuMemAddressFree(va_ranges[i].start, va_ranges[i].sz);
                }
                va_ranges.clear();
            }
        }
        // Assuming everything went well, update everything
        if (status == CUDA_SUCCESS) {
            Range r;
            d_p = new_ptr;
            reserve_sz = aligned_sz;
            r.start = new_ptr;
            r.sz = aligned_sz;
            va_ranges.push_back(r);
        }
    }
    else {
        Range r;
        r.start = new_ptr;
        r.sz = aligned_sz - reserve_sz;
        va_ranges.push_back(r);
        if (d_p == 0ULL) {
            d_p = new_ptr;
        }
        reserve_sz = aligned_sz;
    }

    return status;
}

CUresult
VectorMemMap::grow(size_t new_sz)
{
    CUresult status = CUDA_SUCCESS;
    CUmemGenericAllocationHandle handle;
    if (new_sz <= alloc_sz) {
        return CUDA_SUCCESS;
    }

    const size_t size_diff = new_sz - alloc_sz;
    // Round up to the next chunk size
    const size_t sz = ((size_diff + chunk_sz - 1) / chunk_sz) * chunk_sz;

    if ((status = reserve(alloc_sz + sz)) != CUDA_SUCCESS) {
        return status;
    }

    if ((status = cuMemCreate(&handle, sz, &prop, 0)) == CUDA_SUCCESS) {
        if ((status = cuMemMap(d_p + alloc_sz, sz, 0ULL, handle, 0ULL)) == CUDA_SUCCESS) {
            if ((status = cuMemSetAccess(d_p + alloc_sz, sz, &accessDesc, 1ULL)) == CUDA_SUCCESS) {
                handles.push_back(handle);
                handle_sizes.push_back(sz);
                alloc_sz += sz;
            }
            if (status != CUDA_SUCCESS) {
                (void)cuMemUnmap(d_p + alloc_sz, sz);
            }
        }
        if (status != CUDA_SUCCESS) {
            (void)cuMemRelease(handle);
        }
    }

    return status;
}

}
