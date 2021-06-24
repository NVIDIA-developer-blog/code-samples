/* Copyright (c) 2020-2021, NVIDIA CORPORATION. All rights reserved.
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
#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <string>
#include <vector>
#include <assert.h>

#define AlignSize(x,y) (x + y - 1) & ~(y - 1)
#define TWO_MB 2 * 1024 * 1024
#define SIXTY_FOUR_KB 64 * 1024
#define FOUR_KB 4 * 1024

#define CUDA_CHECK(status) \
  if (status != cudaSuccess) \
  { \
    printf("%s:%d CudaError: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
    assert(0); \
  }

enum KernelOp {
  READ,
};

enum UVMBehavior {
  PAGE_FAULT,
  ZERO_COPY,
  PREFETCH_ONCE_AND_HINTS,
  STRIPE_GPU_CPU,
};

enum MemoryAccess {
  STREAMING,
  BLOCK_STREAMING,
  RANDOM_WARP    // random page per warp, coalseced within warp
};

template <typename T> __device__ T myrand(T i);

// from wikipedia - glibc LCG constants
// x_n+1 = (a*x_n + c) mod m
// a = 1103515245
// m = 2^31
// c  = 12345
template<>
__device__
__forceinline__ uint64_t myrand(uint64_t x)
{
  uint64_t a = 1103515245;
  uint64_t m = (uint64_t)0x1 << 31;
  uint64_t c = 12345;

  return ((a * x + c) % m);
}

template<typename data_type>
__global__ void read_thread(data_type *ptr, const size_t size)
{
  size_t n = size / sizeof(data_type);
  data_type accum = 0;

  for(size_t tid = threadIdx.x + blockIdx.x * blockDim.x; tid < n;
        tid += blockDim.x * gridDim.x)
    accum += ptr[tid];

  if (threadIdx.x == 0)
    ptr[0] = accum;
}

// lock-step block sync version - yield better performance
template<typename data_type>
__global__ void read_thread_blocksync(data_type *ptr, const size_t size)
{
    size_t n = size / sizeof(data_type);
    data_type accum = 0;    // ToDo: check PTX that accum is not optimized out

    size_t tid = threadIdx.x + blockIdx.x * blockDim.x;
    while (1) {
      if ((tid - threadIdx.x) > n) {
        break;
      }
      if (tid < n)
        accum += ptr[tid];
      tid += (blockDim.x * gridDim.x);
      __syncthreads();
    }
    if (threadIdx.x == 0)
      ptr[0] = accum;
}

template<typename data_type>
__global__ void read_thread_blockCont(data_type *ptr, const size_t size)
{
  size_t n = size / sizeof(data_type);
  data_type accum = 0;

  size_t elements_per_block = ((n + (gridDim.x - 1)) / gridDim.x) + 1;
  size_t startIdx = elements_per_block * blockIdx.x;

  for (size_t rid = threadIdx.x; rid < elements_per_block; rid += blockDim.x) {
    if ((rid + startIdx) < n)
      accum += ptr[rid + startIdx];
  }

  if (threadIdx.x == 0)
    ptr[0] = accum;
}

// lock-step block sync version - yield better performance
template<typename data_type>
__global__ void read_thread_blockCont_blocksync(data_type *ptr, const size_t size)
{
  size_t n = size / sizeof(data_type);
  data_type accum = 0;

  size_t elements_per_block = ((n + (gridDim.x - 1)) / gridDim.x) + 1;
  size_t startIdx = elements_per_block * blockIdx.x;

  size_t rid = threadIdx.x + startIdx;
  while (1) {
    if ((rid - threadIdx.x - startIdx) > elements_per_block) {
      break;
    }

    if (rid < n) {
       accum += ptr[rid];
    }
    rid += blockDim.x;
    __syncthreads();
  }

  if (threadIdx.x == 0)
    ptr[0] = accum;
}

template<typename data_type>
__global__ void cta_random_warp_streaming_read(data_type *ptr, const size_t size, size_t num_pages,
                                                size_t page_size)
{
  size_t n = size / sizeof(data_type);
  int loop_count = n / (blockDim.x * gridDim.x);

  size_t dtype_per_page = page_size / sizeof(data_type);
  size_t lane0_idx_mod = dtype_per_page - warpSize;   // so that warp doesnt overshoot page boundary

  int lane_id = threadIdx.x & 31;
  uint64_t idx = threadIdx.x + blockIdx.x * blockDim.x;
  data_type accum = 0;

  uint64_t nRandom = myrand(idx);   // seed
  for (int i = 0; i < loop_count; i++) {
    nRandom = myrand(nRandom);
    uint64_t page_number = nRandom % num_pages;

    // warp lane 0 broadcast page number to all other warp lanes
    page_number = __shfl_sync(0xffffffff, page_number, 0);

    // coalesced 128 byte access within page - not aligned
    // maybe access two cache lines instead of one
    uint64_t page_idx = nRandom % lane0_idx_mod;
    page_idx = __shfl_sync(0xffffffff, page_idx, 0);
    page_idx += lane_id;

    accum += ptr[page_number * dtype_per_page + page_idx];
    
    idx += blockDim.x * gridDim.x;
  }

  if (threadIdx.x == 0)
    ptr[0] = accum;
}

int main(int argc, char *argv[]) {

  KernelOp        k_op = READ;
  UVMBehavior     uvm_behavior = PAGE_FAULT;
  MemoryAccess    memory_access = STREAMING;
  float           oversubscription_factor = 1.0f; // 1.0 - 100%
  size_t          page_size = TWO_MB;
  int loop_count = 3;
  int num_gpus = 0;
  int block_size = 128;
  CUDA_CHECK(cudaGetDeviceCount(&num_gpus));

  std::string header_string = "";

  int cur_pos = 1;
  while (cur_pos < argc) {
    std::string flag = argv[cur_pos++];
    if (flag == "-m") {
      // uvm mode
      // default (page-fault), prefetch, stripe_gpu_cpu, STRIPE_GPU_GPU
      std::string flag_val = argv[cur_pos++];
      if (flag_val == "prefetch_once_and_hints")
        uvm_behavior = PREFETCH_ONCE_AND_HINTS;
      else if (flag_val == "stripe_gpu_cpu")
        uvm_behavior = STRIPE_GPU_CPU;
      else if (flag_val == "zero_copy")
        uvm_behavior = ZERO_COPY;
      else
        uvm_behavior = PAGE_FAULT;
    }
    else if (flag == "-a") {
      // test
      std::string flag_val = argv[cur_pos++];
      if (flag_val == "streaming")
        memory_access = STREAMING;
      else if (flag_val == "block_streaming")
        memory_access = BLOCK_STREAMING;
      else if (flag_val == "random_warp")
        memory_access = RANDOM_WARP;
    }
    else if (flag == "-o") {
      std::string flag_val = argv[cur_pos++];
      if (flag_val == "read")
        k_op = READ;
    }
    else if (flag == "-p") {
      oversubscription_factor = (float)std::atof(argv[cur_pos++]);
    }
    else if (flag == "-s") {
      std::string flag_val = argv[cur_pos++];
      if (flag_val == "2M")
        page_size = TWO_MB;
      else if (flag_val == "64K")
        page_size = SIXTY_FOUR_KB;
      else if (flag_val == "4K")
        page_size = FOUR_KB;
      else {
        printf("Set valid page size: 2M/64K/4K\n");
        exit(-1);
      }
    }
    else if (flag == "-lc") {
      loop_count = std::atoi(argv[cur_pos++]);
      header_string += "loop_count=";
    }
    else if (flag == "-blocksize") {
      block_size = std::atoi(argv[cur_pos++]);
    }
  }

  // log string
  header_string += "Read,";
  std::string mode_str;
  if (uvm_behavior == PAGE_FAULT)
    mode_str = "Page_Fault,";
  else if (uvm_behavior == ZERO_COPY)
    mode_str = "Zero_copy,";
  else if (uvm_behavior == STRIPE_GPU_CPU)
    mode_str = "stripe_gpu_cpu,";
  else if (uvm_behavior == PREFETCH_ONCE_AND_HINTS)
    mode_str = "prefetch_once_and_hints,";
  header_string += mode_str;

  std::string access_str;
  if (memory_access == STREAMING)
    access_str = "streaming,";
  else if (memory_access == BLOCK_STREAMING)
    access_str = "block_streaming,";
  else if (memory_access == RANDOM_WARP)
    access_str = "random_warp,";  
  header_string += access_str;
  header_string += std::to_string(oversubscription_factor);
  header_string += ",";
  
  if (page_size == TWO_MB)
    header_string += "2MB,";
  else if (page_size == SIXTY_FOUR_KB)
    header_string += "64KB,";
  else if (page_size == FOUR_KB)
  header_string += "4KB,";

  header_string += "blocksize=";
  header_string += std::to_string(block_size);
  header_string += ",";


  header_string += "loop_count=";
  header_string += std::to_string(loop_count);

  // determine cudaMallocManaged size
  int current_device = 0;
  CUDA_CHECK(cudaSetDevice(current_device));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, current_device));
  bool is_P9 = (prop.pageableMemoryAccessUsesHostPageTables == 1);

  size_t allocation_size = (size_t)(oversubscription_factor * prop.totalGlobalMem);

  void *blockMemory = nullptr;
  if (memory_access != STREAMING && uvm_behavior) {
    // reduce test working memory
    // cudaMalloc 2/3 GPU
    size_t cudaMallocSize = AlignSize(size_t(prop.totalGlobalMem * 0.67), page_size);
    allocation_size = AlignSize(size_t(prop.totalGlobalMem * 0.33 * oversubscription_factor),
                                  page_size);
    CUDA_CHECK(cudaMalloc(&blockMemory, cudaMallocSize));
  }
  // pad allocation to page_size
  allocation_size = AlignSize(allocation_size, page_size);
  size_t num_pages = allocation_size / page_size;

  size_t avail_phy_vidmem = 0, total_phy_vidmem = 0;
  // allocate memory - add hints etc as needed
  void *uvm_alloc_ptr = NULL;

  // For P9 we need to allocate and free in-benchmark loop
  // as evicted memory has remote mappings don't trigger a page-fault
  if (!(is_P9 && uvm_behavior == PAGE_FAULT)) {
    CUDA_CHECK(cudaMallocManaged(&uvm_alloc_ptr, allocation_size));
    CUDA_CHECK(cudaMemGetInfo(&avail_phy_vidmem, &total_phy_vidmem));
    // populate pages on GPU
    CUDA_CHECK(cudaMemPrefetchAsync(uvm_alloc_ptr, allocation_size, current_device));
  }

  // P9 need more state space on vidmem - size in MB
  size_t state_space_size = (prop.pageableMemoryAccessUsesHostPageTables == 1) ? 320 : 128;
  size_t permissible_phys_pages_count = avail_phy_vidmem / page_size;
  permissible_phys_pages_count -= (state_space_size * 1024 * 1024 / page_size);

  dim3 block(block_size,1,1);
  dim3 grid((prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor) / block.x, 1, 1);
  int num_blocks_per_sm = 1;  // placeholder value

  cudaStream_t task_stream;
  CUDA_CHECK(cudaStreamCreate(&task_stream));

  cudaEvent_t startE, stopE;
  CUDA_CHECK(cudaEventCreate(&startE));
  CUDA_CHECK(cudaEventCreate(&stopE));

  float kernel_time = 0.0f;
  float accum_kernel_time = 0.0f;
  float accum_bw = 0.0f;

  for (int itr = 0; itr < loop_count; itr++) {
    if (is_P9 && uvm_behavior == PAGE_FAULT) {
      CUDA_CHECK(cudaMallocManaged(&uvm_alloc_ptr, allocation_size));
    }
    //  prefetch to CPU as starting point
    if (uvm_behavior != PREFETCH_ONCE_AND_HINTS)
      CUDA_CHECK(cudaMemPrefetchAsync(uvm_alloc_ptr, allocation_size, cudaCpuDeviceId,
                                        task_stream));

    switch(uvm_behavior) {
      case STRIPE_GPU_CPU:
      {
        // distribute pages across GPU0 and CPU
        // get page-split ratios
        float cpu_factor = oversubscription_factor - 1.0;
        if (cpu_factor < 0.0f)
          cpu_factor = 0.0f; 
        int mod_zero_devId = cudaCpuDeviceId;
        int flip_devId = current_device;
        int mod_scale = num_pages;
        if (cpu_factor > 1.0) {
          mod_zero_devId = current_device;
          flip_devId = cudaCpuDeviceId;
          mod_scale = int(std::round(oversubscription_factor));
        }
        else if (cpu_factor != 0.0f) {
          mod_scale = int(std::round(oversubscription_factor / cpu_factor));
        }
        int gpu_page_count = 0, cpu_page_count = 0;
        void *running_ptr = uvm_alloc_ptr;
      
        for (int i = 0; i < num_pages; i++) {
          int device = flip_devId;
          if ((i % mod_scale) == 0 && i != 0)
            device = mod_zero_devId;

          if (gpu_page_count == permissible_phys_pages_count)
            device = cudaCpuDeviceId;

          CUDA_CHECK(cudaMemPrefetchAsync(running_ptr, page_size, device, task_stream));

          if (device == cudaCpuDeviceId)
            cpu_page_count++;
          else
            gpu_page_count++;

          if (itr == 0) {
            CUDA_CHECK(cudaMemAdvise(running_ptr, page_size, cudaMemAdviseSetPreferredLocation,
                                      device));

            if (device == cudaCpuDeviceId)
              CUDA_CHECK(cudaMemAdvise(running_ptr, page_size, cudaMemAdviseSetAccessedBy,
                                        current_device));
          }
          running_ptr = reinterpret_cast<void*>((size_t)running_ptr + page_size);
        }
      }
      break;
      case PREFETCH_ONCE_AND_HINTS:
      {
      // in oversubscrription this is going to over-flow back to sysmem
        if (itr == 0) {
          CUDA_CHECK(cudaMemAdvise(uvm_alloc_ptr, allocation_size, cudaMemAdviseSetAccessedBy,
                                    current_device));
          CUDA_CHECK(cudaMemPrefetchAsync(uvm_alloc_ptr, allocation_size, current_device,
                                            task_stream));
        }
      }
      break;
      case ZERO_COPY:
      {
        if (itr == 0) {
          CUDA_CHECK(cudaMemAdvise(uvm_alloc_ptr, allocation_size,
                                    cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
          CUDA_CHECK(cudaMemAdvise(uvm_alloc_ptr, allocation_size,
                                    cudaMemAdviseSetAccessedBy, current_device));
        }
      }
      default:
        break;
      }

      //CUDA_CHECK(cudaDeviceSynchronize());
      // timer start
      CUDA_CHECK(cudaEventRecord(startE, task_stream));

      // run read/write kernel for streaming/random access
      if (k_op == READ) {
        if (memory_access == STREAMING) {
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm,
                                                read_thread_blocksync<float>, block.x, 0);
          grid.x = prop.multiProcessorCount * num_blocks_per_sm;
          read_thread_blocksync<float><<<grid, block, 0, task_stream>>>((float*)uvm_alloc_ptr,
                                                                allocation_size);
        }
        else if (memory_access == BLOCK_STREAMING) {
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm,
                                                read_thread_blockCont_blocksync<float>, block.x, 0);
          grid.x = prop.multiProcessorCount * num_blocks_per_sm;
          read_thread_blockCont_blocksync<float><<<grid, block, 0, task_stream>>>((float*)uvm_alloc_ptr,
                                                                allocation_size);
        }
        else if (memory_access == RANDOM_WARP) {
          cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks_per_sm,
                                            cta_random_warp_streaming_read<float>, block.x, 0);
          grid.x = prop.multiProcessorCount * num_blocks_per_sm;
          cta_random_warp_streaming_read<float><<<grid, block, 0, task_stream>>>(
                                                (float*)uvm_alloc_ptr, allocation_size,
                                                  num_pages, page_size);
        }
      }

      // timer stop
      CUDA_CHECK(cudaEventRecord(stopE, task_stream));
      CUDA_CHECK(cudaEventSynchronize(stopE));
      CUDA_CHECK(cudaEventElapsedTime(&kernel_time, startE, stopE));
      accum_kernel_time += kernel_time;

      float bw_meas = allocation_size / (1024.0f * 1024.0f * 1024.0f) / (kernel_time / 1000.0f );
      accum_bw += bw_meas;

      if (is_P9 && uvm_behavior == PAGE_FAULT) {
        CUDA_CHECK(cudaFree(uvm_alloc_ptr));
      }
    }

    CUDA_CHECK(cudaEventDestroy(startE));
    CUDA_CHECK(cudaEventDestroy(stopE));
    CUDA_CHECK(cudaStreamDestroy(task_stream));
  
    // avg time, comp bw, print numbers, avg bw per run or total run/total sizes??, avg kernel time, avg bw
    printf("%s, %f ms, %f GB/s\n", header_string.c_str(), accum_kernel_time / loop_count, accum_bw / loop_count);

    if (!(is_P9 && uvm_behavior == PAGE_FAULT)) {
      CUDA_CHECK(cudaFree(uvm_alloc_ptr));
    }

    if (blockMemory)
      CUDA_CHECK(cudaFree(blockMemory));
}
