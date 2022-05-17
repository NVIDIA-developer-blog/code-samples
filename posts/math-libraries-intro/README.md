# GEMM Examples

============================

Matrix Multiplication performed using OpenBLAS and cuBLAS.

## Getting Started

============================

This example requires the following packages:

- CUDA Toolkit 11.0
- OpenBLAS 0.2.18
- GCC 5.4.0
- LAPACK 3.6.1

## Running examples

============================

`git clone the repo`

`cd code-samples/posts/math-libraries-intro`

`make`

`./openblas-example`

### Output

=============================

```bash

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

`./cublas-example`

### Output

=============================

```bash

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

## Hardware Specifications

------------------------------

**CPU:**
`Intel(R) Xeon(R) CPU E5-2698 v3 @ 2.30GHz`

**GPU:**
`Tesla V100-PCIE 32GB`
