#include "mergesort.h"

#define MIN( x, y ) ( ((x)<(y)) ? (x):(y) )

__global__ void CUDAsort(double* in, uint64_t count)
{
	uint64_t thread_ID = blockDim.x * ( blockDim.y * ( blockDim.z * ( gridDim.x * (gridDim.y * (blockIdx.z)
		+ blockIdx.y ) + blockIdx.x ) + threadIdx.z ) + threadIdx.y ) + threadIdx.x;

	// tID2 means the initial access point of the input data of each threads
	uint64_t tID2 = thread_ID * 2;

	// Quick swap method while scale = 1
	if (*(in + tID2) > * (in + tID2 + 1) && tID2 + 1 < count)
		CUDAswap(in + tID2);

	return;
}

__global__ void CUDAmerge(double* in, uint64_t count, uint64_t scale)
{
	uint64_t thread_ID = blockDim.x * (blockDim.y * (blockDim.z * (gridDim.x * (gridDim.y * (blockIdx.z)
		+ blockIdx.y) + blockIdx.x) + threadIdx.z) + threadIdx.y) + threadIdx.x;

	// tID2 means the initial access point of the input data of each threads
	uint64_t tID2 = thread_ID * 2;

	// Merge operation
	if (thread_ID % scale == 0 && tID2 + scale < count)
		CUDAcombine(in + tID2, scale, in + tID2 + scale, MIN(scale, count - tID2 - scale));
}

__device__ void CUDAswap(double* in)
{
	double tmp = *in;
	*in = *(in + 1);
	*(in + 1) = tmp;
	return;
}

__device__ void CUDAcombine(double* in1, uint64_t cin1, double* in2, uint64_t cin2)
{
	uint64_t ctmp = cin1 + cin2;
	size_t size = (cin1 + cin2) * sizeof(double);
	double* tmp = (double*)malloc(size);
	double* i1 = NULL;
	double* i2 = NULL;
	uint64_t c1 = cin1;
	uint64_t c2 = cin2;
	uint64_t cc = 0;
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

__host__ void CPUmergesort(double* in, uint64_t count)
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
	uint64_t cout1, cout2;
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

__host__ void CPUsplit(double* input, uint64_t cin, double** out1, uint64_t* cout1, double** out2, uint64_t* cout2)
{
	*cout1 = cin / 2;
	*cout2 = cin - *cout1;
	*out1 = input;
	*out2 = input + *cout1;
	return;
}

__host__ void CPUcombine(double* in1, uint64_t cin1, double* in2, uint64_t cin2)
{
	uint64_t ctmp = cin1 + cin2;
	size_t size = (cin1 + cin2) * sizeof(double);
	double* tmp = (double*)malloc(size);
	double* i1 = NULL;
	double* i2 = NULL;
	uint64_t c1 = cin1;
	uint64_t c2 = cin2;
	uint64_t cc = 0;
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

__host__ void COORPmergesort(double* in, uint64_t count, double rate)
{
	// Timer initialize
	LARGE_INTEGER t1, ts;
	QueryPerformanceFrequency(&ts);
	QueryPerformanceCounter(&t1);

	// Timer start
	fprintf(stderr,"[%lf sec] Timer start!\n", timeStr(t1, ts));

	// 100% CPU Method, without memory exchange between host and device
	if (rate <= 0.0)
	{
		CPUmergesort(in, count);
		fprintf(stderr, "[%lf sec] CPU done! Mergesort done!\n", timeStr(t1, ts));
		return;
	}

	// 100% GPU Method, all the operation will be done in device
	if (rate >= 1.0)
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
		fprintf(stderr, "[%lf sec] cudaMalloc OK!\n",timeStr(t1,ts));
#endif

		// copy the inputs from host to device
		err = cudaMemcpy(d_in, in, size, cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy memory from host to device (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
#ifdef _DEBUG
		fprintf(stderr, "[%lf sec] cudaMemcpy OK!\n", timeStr(t1, ts));
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
		fprintf(stderr, "[%lf sec] cudaDeviceGetAttribute OK!\n", timeStr(t1, ts));
#endif

		// Create CUDA operation structure
		uint64_t needed_thread = (count / 2) + (count % 2);
		dim3 g((uint32_t)needed_thread / max_thread + !(!((uint32_t)needed_thread % max_thread)));
		dim3 b((uint32_t)needed_thread / g.x + !(!((uint32_t)needed_thread % g.x)));
#ifdef _DEBUG
		fprintf(stderr, "[%lf sec] Sorting %llu numbers with a %d x %d x %d grid and %d x %d x %d blocks, %llu threads.\n", timeStr(t1, ts),count,g.z,g.y,g.x,b.z,b.y,b.x,(uint64_t)g.z*g.y*g.x*b.z*b.y*b.x);
#endif

		//Do CUDA mergesort
		CUDAsort <<< g, b >>> (d_in, count);

		// host should idle until all the device work has done
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to synchronize CUDA device (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
#ifdef _DEBUG
		fprintf(stderr, "[%lf sec] CUDAsort OK!\n", timeStr(t1, ts));
#endif

		for (uint64_t scale = 2; scale < count; scale = scale * 2)
		{
			CUDAmerge <<< g, b >>> (d_in, count, scale);

			// host should idle until all the device work has done
			err = cudaDeviceSynchronize();
			if (err != cudaSuccess)
			{
				fprintf(stderr, "Failed to synchronize CUDA device (error code %s)!\n", cudaGetErrorString(err));
				exit(EXIT_FAILURE);
			}
		}

#ifdef _DEBUG
		fprintf(stderr, "[%lf sec] CUDAmerge OK!\n", timeStr(t1, ts));
#endif

		// Copy the inputs from device back to host
		err = cudaMemcpy(in, d_in, size, cudaMemcpyDeviceToHost);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy memory from device to host (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
#ifdef _DEBUG
		fprintf(stderr, "[%lf sec] cudaMemcpy OK!\n", timeStr(t1, ts));
#endif

		// Release CUDA memory
		err = cudaFree(d_in);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to free CUDA memory (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
#ifdef _DEBUG
		fprintf(stderr, "[%lf sec] cudaFree OK!\n", timeStr(t1, ts));
#endif
		fprintf(stderr, "[%lf sec] GPU done! Mergesort done!\n", timeStr(t1, ts));

		return;
	}

	// Cooperative method
	// split the input to host and device, count_d is the number of device ration, count_h is for host.
	uint64_t count_d ,count_h;
	count_d = (uint64_t)((double)count * rate);
	count_h = count - count_d;

	// Child thread preparation
	HANDLE hThread;
	unsigned threadID;

	// Argument packaging, due to the limit of child thread invoke, only 1 argument can be passed to the thread.
	void** parg = new void*[4];
	*parg = in;
	*(parg + 1) = &count_d;
	*(parg + 2) = &t1;
	*(parg + 3) = &ts;


	// Child thread initialization and burst, for device control.
	hThread = (HANDLE)_beginthreadex(NULL, 0, (_beginthreadex_proc_type)gChild, parg, 0,&threadID);

	// Do CPUmergesort
	double* h_in = in + count_d;
	CPUmergesort(h_in, count_h);
	fprintf(stderr, "[%lf sec] CPU done!\n", timeStr(t1, ts));

	// Wait until both of the work has done.
	WaitForSingleObject(hThread, INFINITE);
	CloseHandle(hThread);

	//combine the result between GPU and CPU
	CPUcombine(in, count_d, h_in, count_h);
	fprintf(stderr, "[%lf sec] Mergesort done!\n", timeStr(t1, ts));

	return;
}

void __stdcall gChild(void** parg)
{
	cudaError_t err = cudaSuccess;

	// Argument extraction.
	double* in = (double*)*parg;
	uint64_t count_d = **((uint64_t**)(parg+1));
	size_t size_d = count_d * sizeof(double);

	LARGE_INTEGER t1 = **((LARGE_INTEGER**)(parg + 2));
	LARGE_INTEGER ts = **((LARGE_INTEGER**)(parg + 3));

	// Device memory allocation for the inputs.
	double* d_in = NULL;
	err = cudaMalloc((void**)&d_in, size_d);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device memory (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
#ifdef _DEBUG
	fprintf(stderr, "[%lf sec] cudaMalloc OK!\n", timeStr(t1, ts));
#endif

	// Copy the inputs from host to device
	err = cudaMemcpy(d_in, in, size_d, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy memory from host to device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
#ifdef _DEBUG
	fprintf(stderr, "[%lf sec] cudaMemcpy OK!\n", timeStr(t1, ts));
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
	fprintf(stderr, "[%lf sec] cudaDeviceGetAttribute OK!\n", timeStr(t1, ts));
#endif

	// Create CUDA operation structure
	uint64_t needed_thread = (count_d / 2) + (count_d % 2);
	dim3 g((uint32_t)needed_thread / max_thread + !(!((uint32_t)needed_thread % max_thread)));
	dim3 b((uint32_t)needed_thread / g.x + !(!((uint32_t)needed_thread % g.x)));
#ifdef _DEBUG
	fprintf(stderr, "[%lf sec] Sorting %llu numbers with a %d x %d x %d grid and %d x %d x %d blocks, %llu threads.\n", timeStr(t1, ts), count_d, g.z, g.y, g.x, b.z, b.y, b.x, (uint64_t)g.z * g.y * g.x * b.z * b.y * b.x);
#endif

	//Do CUDA mergesort
	CUDAsort <<< g, b >>> (d_in, count_d);

	// host should idle until all the device work has done
	err = cudaDeviceSynchronize();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to synchronize CUDA device (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
#ifdef _DEBUG
	fprintf(stderr, "[%lf sec] CUDAsort OK!\n", timeStr(t1, ts));
#endif

	for (uint64_t scale = 2; scale < count_d; scale = scale * 2)
	{
		CUDAmerge <<< g, b >>> (d_in, count_d, scale);

		// host should idle until all the device work has done
		err = cudaDeviceSynchronize();
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to synchronize CUDA device (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
	}

#ifdef _DEBUG
	fprintf(stderr, "[%lf sec] CUDAmerge OK!\n", timeStr(t1, ts));
#endif
	// Copy the inputs from device back to host
	err = cudaMemcpy(in, d_in, size_d, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to copy memory from device to host (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
#ifdef _DEBUG
	fprintf(stderr, "[%lf sec] cudaMemcpy OK!\n", timeStr(t1, ts));
#endif

	// Release CUDA memory
	err = cudaFree(d_in);
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to free CUDA memory (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
#ifdef _DEBUG
	fprintf(stderr, "[%lf sec] cudaFree OK!\n", timeStr(t1, ts));
#endif
	fprintf(stderr, "[%lf sec] GPU done!\n", timeStr(t1, ts));
	return;
	// End of the child thread.
}

__host__ double timeStr(LARGE_INTEGER t1, LARGE_INTEGER ts)
{
	LARGE_INTEGER t2;
	QueryPerformanceCounter(&t2);
	return (t2.QuadPart - t1.QuadPart) / (double)(ts.QuadPart);
}