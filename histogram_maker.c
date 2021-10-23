/* 
	This program will calculate an histogram of values that appear in the input (standard input). The calculation proceess involves MPI, OpenMP and CUDA.
	
	Input: integers between 0 to 255.
	Output: the number of occurrences of each integer in the input.

	For more information on this program and how to use instructions please read the README file.
	
	Contributors:
	Omer Lev-Ron
	Sam Media
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include "cuda_api.h"
#include "mpi.h"

enum ranks { ROOT, WORKER };

int** find_histogram(int *arr, int size, int* num_threads)
{
	int **histograms = NULL;

#pragma omp parallel
	{
		int num;
		int thread_num = omp_get_thread_num();
		// allocate only one data structure
#pragma omp single
		{
			*num_threads = omp_get_num_threads();
			histograms = (int**)malloc(*num_threads * sizeof(int*));
		}
		// allocate private histogram for each thread
		histograms[thread_num] = (int*)calloc(INPUT_RANGE, sizeof(int));

#pragma omp for
		for (int i = 0; i < size; i++)
		{
			num = arr[i];
			histograms[thread_num][num] += 1;
		}
	}

	return histograms;
}


int* calc_histogram_with_reduction(int *arr, int size)
{
	int num_threads;
	int* histogram = (int*)calloc(INPUT_RANGE, sizeof(int));
	int sum = 0;

	int** histograms = find_histogram(arr, size, &num_threads);

	#pragma omp parallel for reduction (+:sum)
	for (int i = 0; i < INPUT_RANGE; i++)
	{
		sum = 0;
		for (int j = 0; j < num_threads; j++)
		{
			sum += histograms[j][i];
		}
		histogram[i] = sum;
	}
	// free data structure
	for (int i = 0; i < num_threads; i++)
	{
		free(histograms[i]);
	}
	free(histograms);

	return histogram;
}

int* calc_histogram(int *arr, int size)
{
	int num_threads;
	int* histogram = (int*)calloc(INPUT_RANGE, sizeof(int));
	int** histograms = find_histogram(arr, size, &num_threads);

	#pragma omp for
	for (int i = 0; i < INPUT_RANGE; i++)
	{
		for (int j = 0; j < num_threads; j++)
		{
			histogram[i] += histograms[j][i];
		}
	}
	// free data structure
	for (int i = 0; i < num_threads; i++)
	{
		free(histograms[i]);
	}
	free(histograms);

	return histogram;
}


int main(int argc, char **argv) {
	int histogram[INPUT_RANGE];
	int *arr;
	int *histogram_omp, *histogram_cuda;
	int i, num, count;
	int my_rank, num_procs;
	int omp_size, cuda_size;
	int N, my_size, worker_size;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
	
	if (num_procs != 2) 
	{
       	printf("number of processes must be 2, aborting...\n");
       	MPI_Abort(MPI_COMM_WORLD, __LINE__);
    	}
	
	if (my_rank == ROOT)
	{
		// read numbers from user
		fscanf(stdin, "%d", &N);
		arr = (int*)malloc(sizeof(int) * N);
		for (i = 0; i < N; i++)
		{
			fscanf(stdin, " %d", &num);
			arr[i] = num;
		}
		my_size = N / num_procs;
		worker_size = N - my_size;
		// send the worker his arr size
		MPI_Send(&worker_size, 1, MPI_INT, WORKER, 0, MPI_COMM_WORLD);
		// send the worker his arr data
		MPI_Send(arr + my_size, worker_size, MPI_INT, WORKER, 0, MPI_COMM_WORLD);
		// calc the historgam using OMP with reduction
		histogram_omp = calc_histogram_with_reduction(arr, my_size);
		// get result from the worker
		MPI_Recv(histogram, INPUT_RANGE, MPI_INT, WORKER, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		// add result from worker
		for (i = 0; i < INPUT_RANGE; i++)
			histogram[i] += histogram_omp[i];
		
		// print final result
		for (i = 0; i<INPUT_RANGE; i++)
		{
		 	count = histogram[i];
			if (count != 0)
				printf("%d: %d\n", i, count);
		}
	}

	if (my_rank != ROOT)
	{
		MPI_Recv(&my_size, 1, MPI_INT, ROOT, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		arr = (int*)malloc(sizeof(int)*my_size);
		MPI_Recv(arr, my_size, MPI_INT, ROOT, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		omp_size = my_size / 2;
		cuda_size = my_size - omp_size;
		// calc the historgam using OMP 
		histogram_omp = calc_histogram(arr, omp_size);
		// calc the histogram using CUDA
		histogram_cuda = calc_histogram_on_gpu(arr + omp_size, cuda_size);
		if (histogram_cuda == NULL)
		{
			printf("Failed to calcuate histogram using CUDA, aborting...\n");
			MPI_Abort(MPI_COMM_WORLD, __LINE__);
		}
		// add result from cuda
		for (i = 0; i<INPUT_RANGE; i++)
			histogram_omp[i]+= histogram_cuda[i];
		
		MPI_Send(histogram_omp, INPUT_RANGE, MPI_INT, ROOT, 0, MPI_COMM_WORLD);
		free(histogram_cuda);
	}
	free(arr);
	free(histogram_omp);
	MPI_Finalize();
	return 0;
}