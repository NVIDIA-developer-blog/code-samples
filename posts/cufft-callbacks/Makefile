all: callbacks no_callbacks

callbacks:
	nvcc -ccbin g++ -dc -m64 -o cufft_callbacks.o -c cufft_callbacks.cu
	nvcc -ccbin g++ -m64 -o cufft_callbacks cufft_callbacks.o  -lcufft_static -lculibos

no_callbacks:
	nvcc -ccbin g++ -dc -m64 -o cufft_no_callbacks.o -c cufft_no_callbacks.cu
	nvcc -ccbin g++ -m64 -o cufft_no_callbacks cufft_no_callbacks.o -lcufft -lculibos

test:
	nvcc -ccbin g++ -dc -m64 -o test.o -c test.cu
	nvcc -ccbin g++ -m64 -o test test.o  -lcufft_static -lculibos

clean:
	rm *.o
	rm cufft_callbacks
	rm cufft_no_callbacks