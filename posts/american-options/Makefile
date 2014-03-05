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

