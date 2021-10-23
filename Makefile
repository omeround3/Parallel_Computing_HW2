build:
	mpicxx -fopenmp -c histogram_maker.c -o histogram_maker.o
	nvcc -c cuda_api.cu -o cuda_api.o
	mpicxx -fopenmp -o histogram  histogram_maker.o cuda_api.o /usr/local/cuda-10/targets/aarch64-linux/lib/libcudart_static.a -ldl -lrt

clean:
	rm -f *.o histogram

run:
	mpiexec -n 2 ./histogram < testing_input.txt