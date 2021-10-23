# Parallel Computing - HW2

This program will calculate an histogram of values that appear in the input (standard input). The calculation proceess involves **MPI, OpenMP and CUDA**.
	
- Input: integers between 0 to 255.
- Output: the number of occurrences of each integer in the input.

The program consists of two MPI processes.
- Process 0 will read the input into an array of
Numbers and will send half of the array to the second process. Each process will calculate the histogram of
One of the halves of the array.
    - Process 0 (of MPI) will use OpenMP (without CUDA) to calculate the histogram
his.
- Process 1 (The second process) will send the histogram calculated to process 0 to merge
The same with the histogram that he himself calculated and would write the output.
    - The second MPI process will use both OpenMP and CUDA to calculate its histogram: with OpenMP the histogram of half the input arrays for which it is responsible (which are a quarter of the values in the original input) will be calculated and with the help of CUDA will be calculated
    (the histogram of the remaining part). After calculating these 2 histograms, the second process will merge them into one histogram and the merged histogram will be sent to process 0 (which as mentioned will merge it with the histogram he himself calculated.

## How to run the program
1. Clone this git repository to an Ubuntu 18.04/20.04 machine.
2. Make sure you have `gcc`, `mpicc` and `nvcc` installed. You can check by running the next commands:
    - `gcc --version`
    - `mpicc --version`
    - `nvcc --version`
3. Set the `libcudart_static.a` path in the make file.
4. Use the Makefile to build and run:
    - `make`
    - `make run`

