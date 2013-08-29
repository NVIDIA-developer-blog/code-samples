CC=gcc
CXX=g++
NVCC=nvcc

BINARIES=manual_nvtx compiler_inst_nvtx

all: $(BINARIES)

inst_nvtx.o: inst_nvtx.cpp Makefile
	g++ -fPIC -I${CUDA_ROOT}/include -c inst_nvtx.cpp

manual_nvtx: manual_nvtx.cu Makefile
	nvcc -DUSE_NVTX -arch=sm_20 -lnvToolsExt -o manual_nvtx manual_nvtx.cu

compiler_inst_nvtx: compiler_inst_nvtx.cu Makefile inst_nvtx.o
	nvcc -Xcompiler -fPIC -Xcompiler -finstrument-functions -arch=sm_20 inst_nvtx.o -ldl -lnvToolsExt -o compiler_inst_nvtx compiler_inst_nvtx.cu 
	
clean:
	rm -f *.o $(BINARIES)
	
run: $(BINARIES)
	nvprof -o compiler_inst_nvtx.nvvp ./compiler_inst_nvtx
	nvprof -o manual_nvtx.nvvp ./manual_nvtx
