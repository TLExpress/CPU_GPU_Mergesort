#include "mergesort.h"
#include <stdio.h>
#include <stdlib.h>
#include <cooperative_groups.h>

#define thread_ID  (blockDim.x * blockIdx.x + threadIdx.x)
// DISCARDED CODES
//((blockIdx.z*gridDim.x*gridDim.y+blockIdx.y*gridDim.x+blockIdx.x)*blockDim.x*blockDim.y*blockDim.z+threadIdx.z*blockDim.x*blockDim.y+threadIdx.y*blockDim.x+threadIdx.x)

__global__ void CUDAmergesort(double* in, unsigned __int64 count)
{
	unsigned __int64 tID2 = thread_ID * 2; // tID2 means the initial point of the input data of each thread
	if (*(in + tID2) > * (in + tID2 + 1) && tID2 + 1 < count)
		CUDAswap(in + tID2);
	__syncthreads();

	// Merge operation
	unsigned __int64 scale;
	for (scale = 2; scale < count; scale = scale * 2)
	{
		if (thread_ID % scale == 0 && tID2 + scale < count)
		{
			CUDAcombine(in + (unsigned __int64)tID2, scale, in + (unsigned __int64)tID2 + scale, scale < (count - (unsigned __int64)tID2 - scale) ? scale : (count - (unsigned __int64)tID2 - scale));
		}
		__syncthreads();
	}
	__syncthreads();
	return;
}

__device__ void CUDAswap(double* in)
{
	double tmp = *in;
	*in = *(in + 1);
	*(in + 1) = tmp;
	return;
}

__device__ void CUDAcombine(double* in1, unsigned __int64 cin1, double* in2, unsigned __int64 cin2)
{
	unsigned __int64 ctmp = cin1 + cin2;
	size_t size = (cin1 + cin2) * sizeof(double);
	double* tmp = (double*)malloc(size);
	double* i1 = NULL;
	double* i2 = NULL;
	unsigned __int64 c1 = cin1;
	unsigned __int64 c2 = cin2;
	unsigned __int64 cc = 0;
	for (i1 = in1, i2 = in2; c1 != 0 || c2 != 0; cc++)
	{
		if ((*i1 < *i2 && c1 != 0) || c2 == 0)
		{
			*(tmp + cc) = *i1;
			i1++;
			c1--;
		}
		else
		{
			*(tmp + cc) = *i2;
			i2++;
			c2--;
		}
	}
	for (cc = 0; cc < ctmp; cc++)
		*(in1 + cc) = *(tmp + cc);
	free(tmp);
	return;
}

__host__ void CPUmergesort(double* in, unsigned __int64 count)
{
	if (count == 1)
		return;
	if (count == 2)
	{
		if (*in > * (in + 1))
			CPUswap(in);
		return;
	}
	double* out1, * out2;
	unsigned __int64 cout1, cout2;
	CPUsplit(in, count, &out1, &cout1, &out2, &cout2);
	CPUmergesort(out1, cout1);
	CPUmergesort(out2, cout2);
	CPUcombine(out1, cout1, out2, cout2);
	return;
}

__host__ void CPUswap(double* in)
{
	double tmp = *in;
	*in = *(in + 1);
	*(in + 1) = tmp;
	return;
}

__host__ void CPUsplit(double* input, unsigned __int64 cin, double** out1, unsigned __int64* cout1, double** out2, unsigned __int64* cout2)
{
	*cout1 = cin / 2;
	*cout2 = cin - *cout1;
	*out1 = input;
	*out2 = input + *cout1;
	return;
}

__host__ void CPUcombine(double* in1, unsigned __int64 cin1, double* in2, unsigned __int64 cin2)
{
	unsigned __int64 ctmp = cin1 + cin2;
	size_t size = (cin1 + cin2) * sizeof(double);
	double* tmp = (double*)malloc(size);
	double* i1 = NULL;
	double* i2 = NULL;
	unsigned __int64 c1 = cin1;
	unsigned __int64 c2 = cin2;
	unsigned __int64 cc = 0;
	for (i1 = in1, i2 = in2; c1 != 0 || c2 != 0; cc++)
	{
		if ((*i1 < *i2 && c1 != 0) || c2 == 0)
		{
			*(tmp + cc) = *i1;
			i1++;
			c1--;
		}
		else
		{
			*(tmp + cc) = *i2;
			i2++;
			c2--;
		}
	}
	for (cc = 0; cc < ctmp; cc++)
		*(in1 + cc) = *(tmp + cc);
	free(tmp);
	return;
}

__host__ void COORPmergesort(double* in, unsigned __int64 count, double rate)
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
		printf("cudaDeviceGetAttribute OK!\n");
#endif

		// Do CUDAmergesort
		CUDAmergesort <<< 1 + count / 2 / max_thread, count / 2 / (1 + count / max_thread)+ (count%2) >>> (d_in, count);
		cudaDeviceSynchronize();

		// Copy the inputs from device back to host
		err = cudaMemcpy(in, d_in, size, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy memory from device to host (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
#ifdef _DEBUG
		printf("cudaMemcpy OK!\n");
#endif

		// Release CUDA memory
		err = cudaFree(d_in);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to free CUDA memory (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
#ifdef _DEBUG
		printf("cudaFree OK!\n");
#endif

		return;
	}

	// 100% CPU Method, without memory exchange between host and device
	else if (rate == 0.0)
	{
		CPUmergesort(in, count);
		return;
	}

	// Coorperative method, the lest combine will be done in host
	cudaError_t err = cudaSuccess;

	unsigned __int64 count_d ,count_h;
	count_d = (double)count * rate;
	count_h = count - count_d;

	size_t size_d = count_d * sizeof(double);

	double* d_in = NULL;
	err = cudaMalloc((void**)&d_in, size_d);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device memory (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
#ifdef _DEBUG
	printf("\ncudaMalloc OK!\n");
#endif

	// Copy the inputs from host to device
	err = cudaMemcpy(d_in, in, size_d, cudaMemcpyHostToDevice);
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
	printf("cudaDeviceGetAttribute OK!\n");
#endif

	// Do CUDAmergesort
	CUDAmergesort <<< 1 + count_d / 2 / max_thread, count_d / 2 / (1 + count_d / max_thread) >>> (d_in, count_d);
	cudaDeviceSynchronize();

	// Do CPUmergesort
	double* h_in = in + count_d;
	CPUmergesort(h_in, count_h);

	// Copy the inputs from device back to host
	err = cudaMemcpy(in, d_in, size_d, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy memory from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
#ifdef _DEBUG
	printf("cudaMemcpy OK!\n");
#endif

	// Release CUDA memory
	err = cudaFree(d_in);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free CUDA memory (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
#ifdef _DEBUG
	printf("cudaFree OK!\n");
#endif

	//combine the result of GPU and CPU
	CPUcombine(in, count_d, h_in, count_h);

	return;
}