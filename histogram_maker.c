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

#include "cudaCalcuations.h"
#include "mpi.h"

#define RANGE 255

enum ranks
{
	ROOT,
	WORKER
};

int **get_histogram(int *threads_amount, int *arr, int size);

void free_struct(int **arr, int size)
{
	for (int i = 0; i < size; i++)
	{
		free(arr[i]);
	}
	free(arr);
}


int *histogram_reduction_calculation(int *arr, int size)
{
	int *histogram = (int *)malloc(RANGE * sizeof(int));
	int threads_amount;
	int sum = 0;

	int **histogram_list = get_histogram(&threads_amount, arr, size);

	#pragma omp parallel for reduction(+ : sum)
	for (int i = 0; i < RANGE; i++)
	{
		sum = 0;
		for (int j = 0; j < threads_amount; j++)
		{
			sum += histogram_list[j][i];
		}
		histogram[i] = sum;
	}
	free_struct(histogram_list, threads_amount);

	return histogram;
}

int **get_histogram(int *threads_amount, int *arr, int size)
{
	int **histogram_list = NULL;

	#pragma omp parallel
	{
		int thread_id = omp_get_thread_num();
		int number;

		#pragma omp single
		{
			*threads_amount = omp_get_num_threads();
			histogram_list = (int **)malloc(*threads_amount * sizeof(int *));
		}

		histogram_list[thread_id] = (int *)calloc(RANGE, sizeof(int)); /* local instance of histogram for each thread */

		#pragma omp for
		for (int i = 0; i < size; i++)
		{
			number = arr[i];
			histogram_list[thread_id][number] += 1;
		}
	}

	return histogram_list;
}

int *omp_calculation(int *arr, int size)
{
	int threads_amount;
	int *histogram = (int *)calloc(RANGE, sizeof(int));
	int **histogram_list = get_histogram(&threads_amount, arr, size);

	#pragma omp for
	for (int i = 0; i < RANGE; i++)
	{
		for (int j = 0; j < threads_amount; j++)
		{
			histogram[i] += histogram_list[j][i];
		}
	}

	free_struct(histogram_list, threads_amount);
	return histogram;
}

void print_results(int *arr)
{
	int appearances;
	for (int i = 0; i < RANGE; i++)
	{
		appearances = arr[i];
		if (appearances != 0)
			printf("%d: %d\n", i, appearances);
	}
}

int main(int argc, char **argv)
{

	int *omp_histogram, *cuda_histogram, *arr, *half_of_array;
	int proccess_rank, procceses_amount, size_omp, size_cuda, user_total_input, size, size_worker, user_input;
	int histogram[RANGE];

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &proccess_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &procceses_amount);
	double start = MPI_Wtime();

	if (procceses_amount != 2)
	{
		printf("Proccess number must be 2\n");
		MPI_Abort(MPI_COMM_WORLD, __LINE__);
	}

	if (proccess_rank == ROOT)
	{
		fscanf(stdin, "%d", &user_total_input);
		arr = (int *)malloc(sizeof(int) * user_total_input);
		for (int i = 0; i < user_total_input; i++)
		{
			fscanf(stdin, " %d", &user_input);
			arr[i] = user_input;
		}

		size = user_total_input / procceses_amount;
		size_worker = user_total_input - size;
		MPI_Send(&size_worker, 1, MPI_INT, WORKER, 0, MPI_COMM_WORLD);
		
		half_of_array = arr + size;
		MPI_Send(half_of_array, size_worker, MPI_INT, WORKER, 0, MPI_COMM_WORLD);

		omp_histogram = histogram_reduction_calculation(arr, size);
		MPI_Recv(histogram, RANGE, MPI_INT, WORKER, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		for (int i = 0; i < RANGE; i++)
			histogram[i] += omp_histogram[i];

		print_results(histogram);
		
		double end = MPI_Wtime();
		printf("Execution time is %f\n", end - start ); 
	}
	else
	{
		MPI_Recv(&size, 1, MPI_INT, ROOT, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		
		arr = (int *)malloc(sizeof(int) * size);
		MPI_Recv(arr, size, MPI_INT, ROOT, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		size_omp = size / 2;
		size_cuda = size - size_omp;
		half_of_array = arr + size_omp;
		cuda_histogram = cuda_calculation(half_of_array, size_cuda);
		omp_histogram = omp_calculation(arr, size_omp);

		if (omp_histogram == NULL)
		{
			printf("OpenMP calculation Failed.\n");
			MPI_Abort(MPI_COMM_WORLD, __LINE__);
		}
		if (cuda_histogram == NULL)
		{
			printf("CUDA calculation Failed.\n");
			MPI_Abort(MPI_COMM_WORLD, __LINE__);
		}
		
		for (int i = 0; i < RANGE; i++)
			omp_histogram[i] += cuda_histogram[i];

		MPI_Send(omp_histogram, RANGE, MPI_INT, ROOT, 0, MPI_COMM_WORLD);
		free(cuda_histogram);
	}
	
	free(omp_histogram);
	free(arr);

	MPI_Finalize();

	return 0;
}