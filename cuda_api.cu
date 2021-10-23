

#include <cuda_runtime.h>
#include "cuda_api.h"
#include <stdio.h>


#define MAXIMUM_THREADS 256

__global__ void histogram_calculator(int* arr, int* results, int size)
{
	extern __shared__ int hist_list[];
	int number;
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if(index < size)
	{
		number = arr[index];
		atomicAdd(&(hist_list[number]), 1);
	}
	__syncthreads();

	atomicAdd(&(results[threadIdx.x]), hist_list[threadIdx.x]);
}

/* Free memory from GPU */
void free_memory(int* arr, int* hist_list)
{
	cudaFree(hist_list);
	cudaFree(arr);
}

/* Get the number of blocks per grid */
int get_blocks_number(int block_size)
{
	int reminder, blocks, sum_of_blocks;

	blocks = block_size / MAXIMUM_THREADS;

	if (block_size % MAXIMUM_THREADS != 0)
		reminder = 1;
	else
		reminder = 0;
		
	sum_of_blocks = blocks + reminder;

	return sum_of_blocks;
}




int* calc_histogram_on_gpu(int* arr, int size)
{
	
	cudaError_t cuda_status;
	bool error_flag = false;
	char* error_msg = "";
	int* histogram = (int*)calloc(MAXIMUM_THREADS, sizeof(int));
	int blocks_num;
	int* arr_int = 0;
	int* hist_cuda = 0;
	
	/* Allocation to CUDA Device */
	cuda_status = cudaMalloc((void**)&arr_int, sizeof(int) * size);
	if (cuda_status != cudaSuccess )
	{
		error_flag = true;
		error_msg = "[ERROR] cuda Malloc";
	}
	
	cuda_status = cudaMalloc((void**)&hist_cuda, sizeof(int) * MAXIMUM_THREADS);
	if (cuda_status != cudaSuccess && !error_flag)
	{
		error_flag = true;
		error_msg = "[ERROR] cuda Malloc";
	}
	
	/* Copy from Host memory to CUDA Device memory */
	cuda_status = cudaMemcpy(arr_int, arr, sizeof(int) * size, cudaMemcpyHostToDevice);
	if (cuda_status != cudaSuccess && !error_flag)
	{
		error_flag = true;
		error_msg = "[ERROR] cuda MemcpyHostToDevice";
	}
	
	
	cuda_status = cudaMemcpy(hist_cuda, histogram, sizeof(int) * MAXIMUM_THREADS, cudaMemcpyHostToDevice);
	if (cuda_status != cudaSuccess && !error_flag)
	{
		error_flag = true;
		error_msg = "[ERROR] cuda MemcpyHostToDevice";
	}
	
	blocks_num = get_blocks_number(size);
	histogram_calculator<<<blocks_num, MAXIMUM_THREADS, sizeof(int) * MAXIMUM_THREADS>>>(arr_int, hist_cuda, size);
	
	cuda_status = cudaDeviceSynchronize();
	if (cuda_status != cudaSuccess && !error_flag)
	{
		error_flag = true;
		error_msg = "[ERROR] cuda DeviceSynchronize";
	}
	
	cuda_status = cudaMemcpy(histogram, hist_cuda, sizeof(int) * MAXIMUM_THREADS, cudaMemcpyDeviceToHost);
	if (cuda_status !=  cudaSuccess && !error_flag)
	{
		error_flag = true;
		error_msg = "[ERROR] cuda MemcpyDeviceToHost";
	}

	if (error_flag == true) {
		printf("%s \n", error_msg);
		free_memory(arr_int, hist_cuda);
		return NULL;
	}

	free_memory(arr_int, hist_cuda);
	return histogram;
}
