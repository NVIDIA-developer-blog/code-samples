# Unified Memory Oversubscription

Benchmark for UVM oversubscription tests

Applicatiopn build: Execute the provided Makefile to build the executable.
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

-s - Software abstracted page Size for memory striping experiments
2M/64K/4K
Default: 2M

-blocksize - Kernel thread block size
Default: 128

-lc - LoopCount - Benchmark iteration count
integer value
Default: 3
```

## Sample commands (with test description):
`uvm_oversubs -p 2.0 -a streaming -m zero_copy` - Test oversubscription with 2x GPU memory size working set, using zero-copy (data placed in CPU memory and directly accessed), and streaming access pattern (see corresponding developer blog for detail).

`uvm_oversubs -p 0.5 -a block_streaming -m fault` - Test oversubscription with half GPU memory allocated using Unified Memory (`cudaMallocManaged`) and block strided kernel read data with page-fault induced migration.

`uvm_oversubs -p 1.5 -a stripe_gpu_cpu -m random_warp` - Test oversubscription with 1.5x GPU memory working set, with memory pages striped between GPU and CPU. Random warp kernel accesses a different 128 byte region of allocation in each loop iteration.
