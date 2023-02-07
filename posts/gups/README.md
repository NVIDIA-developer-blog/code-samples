## GUPS Benchmark

### How to build the benchmark
Build with Makefile with following options:

`GPU_ARCH=xx` where `xx` is the Compute Capibility of the device(s) being tested (default: 80 90). Users could check the CC of a specific GPU using the tables [here](https://developer.nvidia.com/cuda-gpus#compute). The generated executable (called `gups`) supports both global memory GUPS and shared memory GUPS modes. Global memory mode is the default mode. Please refer to the next section for the runtime option to switch between modes. 

Notes on shared memory GUPS: 
1. Note that for shared memory GUPS, unless if dynamic allocation is forced (see below), only CC 80 and CC 90 are supported, for other CC, the shared memory GUPS code will fall back to dynamic allocation mode.
2. To force dynamic shared memory allocation, build with `DYNAMIC_SHMEM=`. Note that this is NOT recommended and will result in incorrect shared memory GUPS numbers as the kernel becomes instruction bound.

For example: `make GPU_ARCH="70 80" DYNAMIC_SHMEM=` will build the executable `gups`, which supports global memory GUPS and shared memory GUPS with dynamic shared memory allocation, for both CC 70 (e.g., NVIIDA V100 GPU) and CC 80 (e.g., NVIDIA A100 GPU). 

### How to run the benchmark
Note that besides GUPS (updates (loop)), our benchmark code support other random access tests, including reads, writes, reads+writes, and updates (no loop). 
You can choose the benchmark type using the `-t` runtime option. Users may need to fine tune access per element option (`-a`) to achieve the best performance. 
Note that the correctness verification is only available for updates (loop)/default test. 

You could use `./gups -h` to get a list of runtime arguments.
```
Usage:
  -n <int> input data size = 2^n [default: 29]
  -o <int> occupancy percentage, 100/occupancy how much larger the working set is compared to the requested bytes [default: 100]
  -r <int> number of kernel repetitions [default: 1]
  -a <int> number of random accesses per input data element [default:  32 (r, w) or 8 (u, unl, rw) for gmem, 65536 for shmem]
  -t <int> test type (0 - update (u), 1 - read (r), 2 - write (w), 3 - read write (rw), 4 - update no loop (unl)) [default: 0]
  -d <int> device ID to use [default: 0]
  -s <int> enable input in shared memory instead of global memory for shared memory GUPS benchmark if s>=0. The benchmark will use max available shared memory if s=0 (for ideal GUPS conditions this must be done at compile time, check README.md for build options). This tool does allow setting the shmem data size with = 2^s (for s>0), however this will also result in an instruction bound kernel that fails to reach hardware limitations of GUPS. [default: -1 (disabled)]
```

You can also use provided Python script to run multiple tests with a single command and get a CSV report. The default setting of the script run all the random access tests. Run `python run.py --help` for the usage options. 
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

### LICENSE 

`gups.cu` is modified based on `randomaccess.cu` file from [link to Github repository](https://github.com/nattoheaven/cuda_randomaccess). The LICENSE file of the Github repository is preserved as `LICENSE.gups.cu`. 

`run.py` and `Makefile` are implemented from scratch by NVIDIA. For the license information of these two files, please refer to the `LICENSE` file.