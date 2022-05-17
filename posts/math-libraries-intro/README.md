# GEMM Examples

============================

Matrix Multiplication performed using OpenBLAS and cuBLAS.

## Getting Started

============================

### Packages Used

- CUDA Toolkit 11.0
- OpenBLAS 0.2.18

### Hardware Specifications

**CPU:**
Intel(R) Xeon(R) CPU E5-2698 v3 @ 2.30GHz

**GPU:**
Tesla V100-PCIE 32GB

#### Set Environment Variables

`export OPENBLAS_NUM_THREADS=32`

#### Set GPU Clocks

``` bash
scs () {

        module load cuda/11.2.1

        DATE=$(date +"%m%d%y-%H%M%S")
        G_NAME=$(nvidia-smi -i 0 --query-gpu=gpu_name --format=csv,nounits,noheader | sed 's/ /-/g')
        G_CLK=$(nvidia-smi -i 0 --query-gpu=clocks.max.sm --format=csv,nounits,noheader)
        M_CLK=$(nvidia-smi -i 0 --query-gpu=clocks.max.memory --format=csv,nounits,noheader)
        P_LIMIT=$(nvidia-smi -i 0 --query-gpu=power.max_limit --format=csv,nounits,noheader)
        DRIVER=$(nvidia-smi -i 0 --query-gpu=driver_version --format=csv,nounits,noheader)

        sudo nvidia-smi
        sudo nvidia-smi -pm ENABLED
        sudo nvidia-smi --auto-boost-default=0
        sudo nvidia-smi -ac ${M_CLK},${G_CLK}
        sudo nvidia-smi -lgc ${G_CLK},${G_CLK}
        sudo nvidia-smi -pl ${P_LIMIT}
        sudo nvidia-smi -q -d POWER,CLOCK
}
```

## Running Examples

============================

### Build

`git clone the repo`

`cd code-samples/posts/math-libraries-intro`

`make`

### OpenBLAS

#### Run

`./openblas-example`

#### Sample Output

=============================

```text

This example computes real matrix C=alpha*A*B+beta*C using
 OpenBLAS dgemm, where A, B, and  C are matrices and
 alpha and beta are double precision scalars

 Initializing data for matrix multiplication C=A*B for matrix
 A(4092x4092) and matrix B(4092x4092)

 Allocating memory for matrices aligned on 64-byte boundary for better
 performance

 Time Elapsed: 414.35 ms

 Example completed.

```

### cuBLAS

#### Run

`./cublas-example`

#### Sample Output

=============================

```text

 This example computes real matrix C=alpha*A*B+beta*C using
 cuBLAS dgemm, where A, B, and  C are matrices and
 alpha and beta are double precision scalars

 Initializing data for matrix multiplication C=A*B for matrix
 A(4092x4092) and matrix B(4092x4092)

 Allocating memory for matrices aligned on 64-byte boundary for better
 performance

 Computing matrix product using cuBLAS dgemm function

 Computations completed.

 Time Elapsed: 19.80 ms

 Deallocating memory

 Example completed.
```
