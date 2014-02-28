CUDA_ARCH_FLAGS := -arch=sm_35
CC_FLAGS += $(CUDA_ARCH_FLAGS)

EXE = dataElem dataElem_um dataElem_um_c++_1 dataElem_um_c++_2

all: $(EXE)

% : %.cu
	nvcc $< $(CC_FLAGS) $(LIB_FLAGS) -o $@

clean: 
	rm -f $(EXE)
	