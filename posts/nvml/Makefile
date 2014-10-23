ALL: run

nvml_test: Makefile nvml_test.cu matrixMul.cu
	nvcc -I$(GPU_DEPLOYMENT_KIT_ROOT_DIR)/include/nvidia/gdk -lnvidia-ml nvml_test.cu matrixMul.cu -o nvml_test

run: nvml_test
	./nvml_test
	
clean:
	rm -f nvml_test
