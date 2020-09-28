// System include
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// For the CUDA runtime routines
#include <cuda_runtime.h>
#include <helper_cuda.h>

//Functions for CUDA mergesort
#include "mergesort.h"
 

int main(int argc, char** argv)
{
	cudaError_t err = cudaSuccess;

	// Get the max Thread per one block of current device
	int max_thread = 0;
	err = cudaDeviceGetAttribute(&max_thread, cudaDevAttrMaxThreadsPerBlock, 0);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to get device attribute from CUDA (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	// Debug message: shows the max Thread per one block of current device
#ifdef _DEBUG
	printf("cudaDevAttrMaxThreadsPerBlock = %d\n\n", max_thread);
#endif

	// get the number of the inputs
	unsigned __int64 count;
	printf("Input number: ");
	scanf("%llu", &count);

	// allocate and check required memory from main memory for inputs
	size_t size = count * sizeof(double);
	double* h_input = (double*)malloc(size);
	if (h_input == NULL)
	{
		fprintf(stderr, "Failed to allocate host memory!\n");
		exit(EXIT_FAILURE);
	}
#ifdef _DEBUG
	printf("\nMalloc OK!\n\n");
#endif

	// get input
	printf("Inputs: ");
	for(int c0 = 0; c0<count; c0++)
		scanf("%lf", h_input+c0);

	// allocate required memory from device
	double* d_input = NULL;
	err = cudaMalloc((void**)&d_input, size);
	if(err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device memory (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
#ifdef _DEBUG
	printf("\ncudaMalloc OK!\n");
#endif

	// copy the inputs from host to device
	err = cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
	if(err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy memory from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
#ifdef _DEBUG
	printf("cudaMemcpy OK!\n");
#endif
	return 0;
}