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

	// get the number of the inputs
	unsigned __int64 count;
	printf("Input number: ");
	scanf("%llu", &count);

	// allocate and check required memory from main memory for inputs
	size_t size = count * sizeof(double);
	double* input = (double*)malloc(size);
	if (input == NULL)
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
		scanf("%lf", input+c0);

	// get sorting rate (1.0 for 100% device, 0 for 100% host)
	double rate = 0.0;
	printf("Rate: ");
	scanf("%lf", &rate);
	COORPmergesort(input, count, rate);
	
	// output the result
	for (int c0 = 0; c0 < count; c0++)
		printf("%.2lf\n", *(input + c0));

	return 0;
}