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

CC=gcc
CXX=g++
NVCC=nvcc

NVTX_INCLUDE ?= $(CUDA_HOME)/include

BINARIES=manual_nvtx compiler_inst_nvtx

all: $(BINARIES)

inst_nvtx.o: inst_nvtx.cpp Makefile
	g++ -export-dynamic -fPIC -I$(NVTX_INCLUDE) -c inst_nvtx.cpp

manual_nvtx: manual_nvtx.cu Makefile
	nvcc -Xcompiler -export-dynamic -DUSE_NVTX -ldl -o manual_nvtx manual_nvtx.cu

compiler_inst_nvtx: compiler_inst_nvtx.cu Makefile inst_nvtx.o
	nvcc -Xcompiler -export-dynamic -Xcompiler -fPIC -Xcompiler -finstrument-functions inst_nvtx.o -ldl -o compiler_inst_nvtx compiler_inst_nvtx.cu 
	
clean:
	rm -f *.o $(BINARIES)
	
run: $(BINARIES)
	nsys profile --trace=cuda,nvtx -o compiler_inst_nvtx ./compiler_inst_nvtx
	nsys profile --trace=cuda,nvtx -o manual_nvtx ./manual_nvtx
