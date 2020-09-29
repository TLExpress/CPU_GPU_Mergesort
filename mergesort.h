#pragma once
#ifndef _MERGESORT_H
#define _MERGESORT_H
#include <cuda_runtime.h>
#include <helper_cuda.h>

// Device function, will access on GPU
__global__ void CUDAmergesort(double* in, unsigned __int64 count);
__device__ void CUDAcombine(double* in1, unsigned __int64 cin1, double* in2, unsigned __int64 cin2);
__device__ void CUDAswap(double* in);

// Host function, will access on CPU
__host__ void CPUmergesort(double* in, unsigned __int64 count);
__host__ void CPUsplit(double* input, unsigned __int64 cin, double** out1, unsigned __int64* cout1, double** out2, unsigned __int64* cout2);
__host__ void CPUcombine(double* in1, unsigned __int64 cin1, double* in2, unsigned __int64 cin2);
__host__ void CPUswap(double* in);

// Coorperative function, start from CPU, will transport a rate of data to access on GPU
__host__ void COORPmergesort(double* in, unsigned __int64 count, double rate);

#endif