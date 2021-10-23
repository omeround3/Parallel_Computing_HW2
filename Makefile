build:
	mpicxx -fopenmp -c histogram_maker.c -o histogram_maker.o
	nvcc -c cudaCalcuations.cu -o cudaCalcuations.o
	mpicxx -fopenmp -o histogram  histogram_maker.o cudaCalcuations.o /usr/local/cuda-10/targets/aarch64-linux/lib/libcudart_static.a -ldl -lrt

clean:
	rm -f *.o histogram

run:
	mpiexec -n 2 ./histogram < testing_input.txt