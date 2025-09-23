## GUPS Benchmark

The GUPS (Giga Updates Per Second) benchmark measures random memory access performance. This implementation supports two distinct memory modes:
- **Global Memory GUPS**: Measures random access performance to GPU global memory
- **Shared Memory GUPS**: Measures random access performance to GPU shared memory

### How to build the benchmark
Build with Makefile with following options:

`GPU_ARCH=xx` where `xx` is the Compute Capability of the device(s) being tested (default: 80 90). Users could check the CC of a specific GPU using the tables [here](https://developer.nvidia.com/cuda-gpus#compute). The generated executable (called `gups`) supports both global memory GUPS and shared memory GUPS modes.

#### Global Memory Mode (Default)
Global memory mode measures random access performance to the GPU's global memory. This is the default mode and works with all GPU architectures.

#### Shared Memory Mode
Shared memory mode measures random access performance within the GPU's shared memory. There are two allocation methods:

**1. Static Shared Memory (Recommended)**
- Provides optimal performance by allocating shared memory at compile time
- Only supported for CC 80 and CC 90 by default
- For other compute capabilities, the code automatically falls back to dynamic allocation

**2. Dynamic Shared Memory (Not Recommended)**
- Allocates shared memory at runtime
- Results in significantly lower performance as the kernel becomes instruction bound
- To force dynamic allocation: build with `DYNAMIC_SHMEM=`
- Should only be used for testing purposes, not for performance measurements

#### Build Examples

Standard build for A100/H100 (CC 80/90) with static shared memory:
```bash
make GPU_ARCH="80 90"
```

Build with forced dynamic shared memory for V100 (CC 70):
```bash
make GPU_ARCH="70 80" DYNAMIC_SHMEM=
```
This will build the executable `gups`, which supports global memory GUPS and shared memory GUPS with dynamic shared memory allocation, for both CC 70 (e.g., NVIDIA V100 GPU) and CC 80 (e.g., NVIDIA A100 GPU). 

### How to run the benchmark

The benchmark supports multiple random access test types:
- **Updates (loop)** - Default GUPS test with atomic CAS operations
- **Reads** - Random read operations
- **Writes** - Random write operations
- **Reads+Writes** - Combined read and write operations
- **Updates (no loop)** - Single update per location

#### Running Global Memory Tests (Default)

For global memory tests, simply run the executable without the `-s` option:

```bash
# Run default GUPS update test with 2^29 elements
./gups

# Run with custom size (2^30 elements)
./gups -n 30

# Run read test instead of update
./gups -t 1
```

#### Running Shared Memory Tests

For shared memory tests, use the `-s` option:

```bash
# Use maximum available shared memory (recommended for performance testing)
./gups -s 0 -t 0

# Use dynamic allocation with 2^10 elements (not recommended for performance)
./gups -s 10 -t 0
```

**Important Notes:**
- For optimal shared memory performance, use `-s 0` which allocates maximum available shared memory
- Using `-s` with values > 0 forces dynamic allocation and results in suboptimal performance
- Correctness verification is only available for updates (loop) test type

#### Command Line Options

```
Usage:
  -n <int> input data size = 2^n [default: 29]
  -o <int> occupancy percentage, 100/occupancy how much larger the working set is compared to the requested bytes [default: 100]
  -r <int> number of kernel repetitions [default: 1]
  -a <int> number of random accesses per input element [default: 32 (r, w) or 8 (u, unl, rw) for gmem, 65536 for shmem]
  -t <int> test type (0 - update (u), 1 - read (r), 2 - write (w), 3 - read write (rw), 4 - update no loop (unl)) [default: 0]
  -d <int> device ID to use [default: 0]
  -s <int> enable input in shared memory instead of global memory for shared memory GUPS benchmark if s>=0.
           s=0: use max available shared memory (recommended for performance)
           s>0: use 2^s elements with dynamic allocation (not recommended for performance)
           [default: -1 (disabled, use global memory)]
```

#### Using the Python Script for Batch Testing

A Python script is provided to run multiple tests and generate CSV reports. The script can test both global and shared memory modes.

**Example Usage:**

```bash
# Run global memory tests with sizes from 2^29 to 2^31
python3 run.py --input-size-begin 29 --input-size-end 31 --memory-loc global

# Run shared memory tests with maximum available shared memory
python3 run.py --memory-loc shared

# Run shared memory tests with dynamic allocation (sizes 2^10 to 2^14)
# Note: This uses dynamic allocation and will show suboptimal performance
python3 run.py --input-size-begin 10 --input-size-end 14 --memory-loc shared
```

**Script Options:**

```
usage: run.py [-h] [--device-id DEVICE_ID]
              [--input-size-begin INPUT_SIZE_BEGIN]
              [--input-size-end INPUT_SIZE_END] [--occupancy OCCUPANCY]
              [--repeats REPEATS]
              [--test {reads,writes,reads_writes,updates,updates_no_loop,all}]
              [--memory-loc {global,shared}]

Benchmark GUPS. Store results in results.csv file.

optional arguments:
  -h, --help            show this help message and exit
  --device-id DEVICE_ID
                        GPU ID to run the test
  --input-size-begin INPUT_SIZE_BEGIN
                        exponent of the input data size begin range, base is 2
                        (input size = 2^n). [Default: 29 for global GUPS,
                        max_shmem for shared GUPS. Global/shared is controlled
                        by --memory-loc
  --input-size-end INPUT_SIZE_END
                        exponent of the input data size end range, base is 2
                        (input size = 2^n). [Default: 29 for global GUPS,
                        max_shmem for shared GUPS. Global/shared is controlled
                        by --memory-loc
  --occupancy OCCUPANCY
                        100/occupancy is how much larger the working set is
                        compared to the requested bytes
  --repeats REPEATS     number of kernel repetitions
  --test {reads,writes,reads_writes,updates,updates_no_loop,all}
                        test to run
  --memory-loc {global,shared}
                        memory buffer in global memory or shared memory
```

### Performance Considerations

#### Global Memory vs Shared Memory Performance
- **Global Memory**: Measures the GPU's ability to perform random updates across the entire global memory space
- **Shared Memory**: Measures random update performance within the limited shared memory of each streaming multiprocessor (SM)

#### Shared Memory Test Variations
1. **Static Allocation with Maximum Shared Memory (`-s 0`)**:
   - Uses all available shared memory per SM (e.g., ~227KB on H100)
   - Total test size = Number of SMs × Max shared memory per SM
   - Provides meaningful performance metrics for shared memory random access

2. **Dynamic Allocation with Custom Size (`-s n` where n > 0)**:
   - Forces dynamic shared memory allocation
   - Typically results in instruction-bound kernels
   - Performance numbers will be significantly lower and not representative of hardware capabilities
   - Should only be used for functional testing, not performance benchmarking

### LICENSE 

`gups.cu` is modified based on `randomaccess.cu` file from [link to Github repository](https://github.com/nattoheaven/cuda_randomaccess). The LICENSE file of the Github repository is preserved as `LICENSE.gups.cu`. 

`run.py` and `Makefile` are implemented from scratch by NVIDIA. For the license information of these two files, please refer to the `LICENSE` file.
