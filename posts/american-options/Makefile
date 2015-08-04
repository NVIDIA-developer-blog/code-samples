# Copyright (c) 2015, NVIDIA CORPORATION. All rights reserved.
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

###################################################################################################
#
# Paths to modify
#
###################################################################################################

# The CUDA path

CUDA_PATH      ?= /usr/local/cuda-5.5

CUDA_INC_PATH  ?= $(CUDA_PATH)/include
CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib64

# The CUB library.

CUB_INC_PATH   ?= external/cub

###################################################################################################
#
# Compiler options
#
###################################################################################################

CXX_COMMON_FLAGS = -O3 -I$(CUB_INC_PATH)

# C++ compiler

CXX = g++
CXX_FLAGS = $(CXX_COMMON_FLAGS) -Wall -I$(CUDA_INC_PATH) 

# Gencodes.

COMMA  = ,
SPACE := 
SPACE += 

ifdef ($(SM_ARCH))
  NVCC_GENCODES = $(foreach ARCH, $(subst $(COMMA), $(SPACE), $(SM_ARCH)), -gencode=arch=compute_$(ARCH),code=sm_$(ARCH))
else
  NVCC_GENCODES += -gencode=arch=compute_20,code=sm_20
  NVCC_GENCODES += -gencode=arch=compute_35,code=sm_35
endif

# CUDA compiler

NVCC = $(CUDA_PATH)/bin/nvcc
NVCC_FLAGS = $(CXX_COMMON_FLAGS) -m64 $(NVCC_GENCODES)

# The flags/library for the Longstaff-Schwartz code.

LONGSTAFF_SCHWARTZ_FLAGS = $(NVCC_FLAGS) -DWITH_FUSED_BETA
LONGSTAFF_SCHWARTZ_LIBS  = -lcurand

ifeq ($(WITH_CPU_REFERENCE), 1)
  LONGSTAFF_SCHWARTZ_FLAGS += -DWITH_CPU_REFERENCE
  LONGSTAFF_SCHWARTZ_LIBS  += -llapack
endif

ifeq ($(WITH_FULL_W_MATRIX), 1)
  LONGSTAFF_SCHWARTZ_FLAGS += -DWITH_FULL_W_MATRIX
endif

###################################################################################################

BINARIES = bin/longstaff_schwartz_svd_2

###################################################################################################
#
# The rules to build the code
#
###################################################################################################
all:
	$(MAKE) dirs
	$(MAKE) $(BINARIES)
  
dirs:
	if [ ! -d bin ] ; then mkdir bin ; fi
  
bin/longstaff_schwartz_svd_2: longstaff_schwartz_svd_2.cu
	$(NVCC) $(LONGSTAFF_SCHWARTZ_FLAGS) -o $@ $< $(LONGSTAFF_SCHWARTZ_LIBS)

clean:
	rm -rf bin

