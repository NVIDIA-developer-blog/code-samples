# Copyright (c) 1993-2015, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

====================================================
Description document for the GPU-based Jacobi solver
====================================================

Contents:
---------
1)	Application overview
2)	Build instructions
3)	Run instructions
4)	Documentation


=======================
1) Application overview
=======================

This is a distributed Jacobi solver, using GPUs to perform the computation and MPI for halo exchanges.
It uses a 2D domain decomposition scheme to allow for a better computation-to-communication ratio than just 1D domain decomposition.
All sources for the Jacobi solver can be found in the "src" folder.
They have the following roles:

a) 	Jacobi.h 			- the main header, containing configuration parameters and prototypes of the most important functions
b)	Jacobi.c 			- the application entry point
c)	Input.c				- the command-line argument parser
d)	Host.c	 			- the functions covering host-side processing, including the main Jacobi loop and data exchanges
e)	Device.cu			- the device (GPU) kernels and the host wrappers for these kernels
f)	CUDA_Normal_MPI.c	- the functions managing data exchange through normal MPI (i.e. using intermediate host buffers)
g)	CUDA_Aware_MPI.c	- the functions managing data exchange through CUDA-aware MPI (i.e. without intermediate host buffers)

The flow of the application is as follows:

a)	The MPI environment is initialized (and the desired CUDA device is selected, when using CUDA-aware MPI)
b)	The command-line arguments are parsed
c)	Resources (including host and device memory blocks, streams etc.) are initialized
d)	The Jacobi loop is executed; in every iteration, the local block is updated and then the halo values are exchanged; the algorithm
	converges when the global residue for an iteration falls below a threshold, but it is also limited by a maximum number of
	iterations (irrespective if convergence has been achieved or not)
e)	Run measurements are displayed and resources are disposed

The application uses the following command-line arguments:
a)	-t x y		-	mandatory argument for the process topology, "x" denotes the number of processes on the X direction (i.e. per row) and
					"y" denotes the number of processes on the Y direction (i.e. per column); the topology size must always match the number of
					available processes (i.e. the number of launched MPI processes must be equal to x * y)
b)	-d dx dy 	-	optional argument indicating the size of the local (per-process) domain size; if it is omitted, the size will default to
					DEFAULT_DOMAIN_SIZE as defined in "Jacobi.h"
c)	-fs 		-	optional argument indicating that the replacement of the old block with the new one after an update should be performed using
					a fast pointer swap rather than a full block copy (which is the default behavior)
d)	-h | --help	-	optional argument for printing help information; this overrides all other arguments

=====================
2) Build instructions
=====================

To build the application, please ensure that the following are available on the platform:

a) an MPI implementation (with optional support for CUDA-aware MPI if this version of the application is to be built)
b) a CUDA toolkit (preferably, the latest available)

You can build the CUDA-aware MPI an the normal MPI version by calling make in src

cd src
make

To find mpi.h and the CUDA runtime library the provied Makefile relies on CUDA_INSTALL_PATH and MPI_HOME beeing set correctly. If you are using CUDA 5 or newer and running on a device with compute capability 3.0 or 3.5 you should also add GENCODE_SM30 and GENCODE_SM35 to the GENCODE_FLAGS in the Makefile. Also the macro ENV_LOCAL_RANK might need to
be changed in Jacobi.h to handle the GPU affinit properly. It defaults to MV2_COMM_WORLD_LOCAL_RANK which works with MVAPICH2.
The generated binaries can then be found found the bin directory.

===================
3) Run instructions
===================

To run the normal MPI version use: 
mpiexec -np 2 ./jacobi_cuda_normal_mpi -t 2 1

To run the CUDA-aware MPI version depending on the MPI implementation you are using you need to activate the CUDA-aware feature, e.g. for MVAPICH2 use
MV2_USE_CUDA=1 mpiexec -np 2 --exports=MV2_USE_CUDA ./jacobi_cuda_aware_mpi -t 2 1
 
================
4) Documentation
================

Documentation for this project may be generated automatically using Doxygen by calling make doc. A configuration file for this may be found in the "src" folder. If 
Doxygen is not available, the doc folder also contains pregenerated documentation. 

