# SPDX-FileCopyrightText: Copyright (c) 2020-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# The CUDA compiler.
CUDA_HOME ?= /usr/local/cuda

# The compiler.
CXX = $(CUDA_HOME)/bin/nvcc

# Optimization and Debugging
OPTFLAGS ?= -O3

# Set target GPU CC (only sm_80 and sm_90 are currently supported for STATIC_SHMEM)
GPU_ARCH ?= 80 90

# Default to using compile time NSHMEM
DYNAMIC_SHMEM ?= -DSTATIC_SHMEM

# Source files
SRC_FILES = gups.cu

# Object Files
OBJ_FILES = $(SRC_FILES:.cu=.o)

# CU flags
CU_FLAGS = -std=c++14 -Xcompiler -std=c++14 -lineinfo

CU_FLAGS += $(foreach cc,$(GPU_ARCH), \
	       --generate-code arch=compute_$(cc),code=sm_$(cc) )

# CXX flags
CXXFLAGS = $(OPTFLAGS) $(CU_FLAGS) -Xcompiler -Wall $(DYNAMIC_SHMEM)


LINKFLAGS = $(CXXFLAGS)


DEFAULT: gups

all = gups

gups: $(OBJ_FILES)

# Include the dependencies that were created by %.d rule.
#
ifneq ($(MAKECMDGOALS),clean)
-include $(SRC_FILES:.cu=.d)
endif
#

# Prepare file holding dependencies, to be included in this file.
#

%.d: %.cu Makefile
	@set -e; rm -f $@; \
	$(CXX) -DMAKE_DEPEND -M $(CXXFLAGS) $< > $@.$$$$; \
	sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
	rm -f $@.$$$$

%.o: %.cu Makefile
	$(CXX) $(CXXFLAGS) -c $*.cu

$(all):%:
	$(CXX) $(LINKFLAGS) -o $@ $^

clean:
	rm -f $(OBJ_FILES) *.o *.d gups \
	*.d.[0-9][0-9][0-9][0-9][0-9] *.d.[0-9][0-9][0-9][0-9] \
	*.d.[0-9][0-9][0-9] *.d.[0-9][0-9][0-9][0-9][0-9][0-9] *~
