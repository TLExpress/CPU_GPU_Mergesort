#include "mergesort.h"
#include <stdio.h>

__global__ void CUDAmergesort(double* in, __int64 count)
{
	return;
}

__host__ void CPUmergesort(double* in, __int64 count)
{
	return;
}

__host__ void COORPmergesort(double* in, __int64 count, double rate)
{
	// 100% GPU Method, all the operation will be done in device
	if (rate == 1.0)
	{
		cudaError_t err = cudaSuccess;
		size_t size = count * sizeof(double);

		// allocate required memory from device
		double* d_in = NULL;
		err = cudaMalloc((void**)&d_in, size);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to allocate device memory (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
#ifdef _DEBUG
		printf("\ncudaMalloc OK!\n");
#endif

		// copy the inputs from host to device
		err = cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy memory from host to device (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
#ifdef _DEBUG
		printf("cudaMemcpy OK!\n");
#endif

		int max_block = 0;
		int max_thread = 0;

		// Get the max Threads per one block, and max blocks per one multiprocessor of current device
		err = cudaDeviceGetAttribute(&max_thread, cudaDevAttrMaxThreadsPerBlock, 0);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to get device attribute from CUDA (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		err = cudaDeviceGetAttribute(&max_block, cudaDevAttrMaxBlocksPerMultiprocessor, 0);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to get device attribute from CUDA (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
#ifdef _DEBUG
		printf("cudaDeviceGetAttribute OK!\n\n");
#endif
		// Do CUDAmergesort
		CUDAmergesort <<< 1 + count / max_thread, count / (1 + count / max_thread) >>> (d_in, count);

		// copy the inputs from device back to host
		err = cudaMemcpy(in, d_in, size, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy memory from device to host (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		return;
	}

	// 100% CPU Method, without memory exchange between host and device
	else if (rate == 0.0)
	{
		CPUmergesort(in, count);
		return;
	}

	return;
}