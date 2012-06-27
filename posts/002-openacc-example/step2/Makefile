#
#  Copyright 2012 NVIDIA Corporation
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

PGCC     = pgcc
HMPP     = hmpp

CC       ?= gcc
ACC      ?= $(PGCC)

CCFLAGS  ?= -I../common

PGCC_ACCFLAGS ?= -fast -Minfo=accel -acc
PGCC_OMPFLAGS ?= -fast -Minfo -mp

HMPP_FLAGS = --codelet-required

HMPP_ACCFLAGS ?= -std=c99 -O3
HMPP_OMPFLAGS ?= -std=c99 -O3 -fopenmp -lm

ifeq ($(ACC), $(PGCC))
    ACCFLAGS = $(PGCC_ACCFLAGS)
    OMPFLAGS = $(PGCC_OMPFLAGS)
    ACC_COMMAND = $(ACC)
    OMP_COMMAND = $(ACC)
else
    ACCFLAGS = $(HMPP_ACCFLAGS)
    OMPFLAGS = $(HMPP_OMPFLAGS)
    ACC_COMMAND = $(HMPP) $(HMPP_FLAGS) $(CC)
    OMP_COMMAND = $(CC)
endif

BIN =  laplace2d_omp laplace2d_acc

all: $(BIN)

laplace2d_acc: laplace2d.c
	$(ACC_COMMAND) $(CCFLAGS) -o $@ $< $(ACCFLAGS)

laplace2d_omp: laplace2d.c
	$(OMP_COMMAND) $(CCFLAGS) -o $@ $< $(OMPFLAGS) 

clean:
	$(RM) $(BIN) *.hmf*
