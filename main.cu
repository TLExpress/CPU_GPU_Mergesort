// System include
#include <stdlib.h> // For fflush and rand
#include <stdint.h> // For uint64_t
#include <stdio.h> // Standard IO library
#include <time.h> // For srand()

// For the CUDA runtime routines
#include <cuda_runtime.h>
#include <helper_cuda.h>

//Functions for CUDA mergesort
#include "mergesort.h"
 

int main(int argc, char** argv)
{

	// get the number of the inputs
	uint64_t count;
	printf("Input number:\n");
	scanf("%llu", &count);
	fflush(stdin);

	// allocate and check required memory from main memory for inputs
	size_t size = count * sizeof(double);
	double* input = (double*)malloc(size); // Alloc memory to store the unput
	if (input == NULL)
	{
		fprintf(stderr, "Failed to allocate host memory!\n");
		exit(EXIT_FAILURE);
	}
#ifdef _DEBUG
	fprintf(stderr,"\nMalloc OK!\n");
#endif

	// get input
	printf("Inputs:\n");
	for (int c0 = 0; c0 < count; c0++)
	{
		scanf("%lf", input + c0); 
		fflush(stdin);
	}

	// get sorting rate (1.0 for 100% device, 0 for 100% host)
	double rate = 0.0;
	printf("Rate:\n");
	scanf("%lf", &rate);
	fflush(stdin);

	// Do mergesort
	COORPmergesort(input, count, rate);
	
	// output the result
	printf("\nOutputs:\n");
	for (int c0 = 0; c0 < count; c0++)
		printf("%.2lf\n", *(input + c0));
	
	//release the memory of inputs
	free(input);
	return 0;
}