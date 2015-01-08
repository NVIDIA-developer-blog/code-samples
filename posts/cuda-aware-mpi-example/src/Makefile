# =======================================================================
# Copyright 1993-2013 NVIDIA Corporation.  All rights reserved.
#
# Please refer to the NVIDIA end user license agreement (EULA) associated
# with this source code for terms and conditions that govern your use of
# this software. Any use, reproduction, disclosure, or distribution of
# this software and related documentation outside the terms of the EULA
# is strictly prohibited.
# =======================================================================

# Compilers
MPICC=$(PREP) mpicc
NVCC=$(PREP) nvcc

# Flags
CFLAGS=-std=c99 -O3 -march=native -Wall
MPICFLAGS=-I${MPI_HOME}/include
CUDACFLAGS=-I${CUDA_INSTALL_PATH}/include

GENCODE_SM13    := -gencode arch=compute_13,code=sm_13
GENCODE_SM20    := -gencode arch=compute_20,code=sm_20
GENCODE_SM30    := -gencode arch=compute_30,code=sm_30
GENCODE_SM35    := -gencode arch=compute_35,code=\"sm_35,compute_35\"
GENCODE_FLAGS   := $(GENCODE_SM13) $(GENCODE_SM20) $(GENCODE_SM30) $(GENCODE_SM35)

NVCCFLAGS=-O3 $(GENCODE_FLAGS) -Xcompiler -march=native

CUDALDFLAGS=-L${CUDA_INSTALL_PATH}/lib64 -lcudart

# Description of binaries
BINDIR=../bin
JACOBI_CUDA_NORMAL_MPI=$(BINDIR)/jacobi_cuda_normal_mpi
JACOBI_CUDA_AWARE_MPI=$(BINDIR)/jacobi_cuda_aware_mpi
BINARIES=$(JACOBI_CUDA_NORMAL_MPI) $(JACOBI_CUDA_AWARE_MPI)

# Commands
all: $(BINARIES)

device.o: Jacobi.h Device.cu Makefile
	$(NVCC) $(MPICFLAGS) $(NVCCFLAGS) -c Device.cu -o device.o

input.o: Jacobi.h Input.c Makefile
	$(MPICC) $(MPICFLAGS) $(CUDACFLAGS) $(CFLAGS) -c Input.c -o input.o

jacobi.o: Jacobi.h Jacobi.c Makefile
	$(MPICC) $(MPICFLAGS) $(CUDACFLAGS) $(CFLAGS) -c Jacobi.c -o jacobi.o

host.o: Jacobi.h Host.c Makefile
	$(MPICC) $(MPICFLAGS) $(CFLAGS) -c Host.c -o host.o

cuda_normal_mpi.o: Jacobi.h CUDA_Normal_MPI.c Makefile
	$(MPICC) $(MPICFLAGS) $(CFLAGS) $(CUDACFLAGS) -c CUDA_Normal_MPI.c -o cuda_normal_mpi.o

cuda_aware_mpi.o: Jacobi.h CUDA_Aware_MPI.c Makefile
	$(MPICC) $(MPICFLAGS) $(CFLAGS) $(CUDACFLAGS) -c CUDA_Aware_MPI.c -o cuda_aware_mpi.o

$(JACOBI_CUDA_NORMAL_MPI): jacobi.o input.o host.o device.o cuda_normal_mpi.o Makefile
	mkdir -p $(BINDIR)
	$(MPICC) $(CUDALDFLAGS) -o $(JACOBI_CUDA_NORMAL_MPI) jacobi.o input.o host.o device.o  cuda_normal_mpi.o  
	
$(JACOBI_CUDA_AWARE_MPI): jacobi.o input.o host.o device.o cuda_aware_mpi.o Makefile
	mkdir -p $(BINDIR)
	$(MPICC) $(CUDALDFLAGS) -o $(JACOBI_CUDA_AWARE_MPI) jacobi.o input.o host.o device.o cuda_aware_mpi.o  

doc: CUDA_Normal_MPI.c CUDA_Aware_MPI.c Device.cu Host.c Input.c Makefile Jacobi.c Jacobi.h Jacobi.doxygen
	doxygen Jacobi.doxygen
	
clean:
	rm -rf *.o *~ $(BINARIES)
