# Unified Memory Oversubscription

Benchmark for UVM oversubscription tests

Build command: `nvcc uvm_oversubs.cu -gencode arch=compute_70,code=sm_70 -o uvm_oversubs`
## Command line options

```
-m - UVM mode
page_fault/zero_copy/prefetch_once_and_hints/stripe_gpu_cpu
Default: page_fault

-a - Access Pattern
streaming/block_streaming/random_warp
Default: streaming

-p - Oversubscription factor
Float value.
Eg: 1.1 - 110% GPU allocation
Default: 1.0

-s - Page Size
2M/64K/4K
Default: 2M

-lc - LoopCount - Benchmark iteration count
integer value
```