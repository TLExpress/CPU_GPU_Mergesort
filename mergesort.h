#pragma once
#ifndef _MERGESORT_H
#define _MERGESORT_H
// System include
#include <stdint.h> // for uint64_t and uint32_t
#include <stdio.h> // Standard IO library
#include <process.h> // for multithread programing
#include <Windows.h> // Windows API
#include <time.h>

// For the CUDA runtime routines
#include <cuda_runtime.h>
#include <helper_cuda.h>

// Device function, will access on GPU
__global__ void CUDAsort(double* in, unsigned __int64 count);
__global__ void CUDAmerge(double* in, unsigned __int64 count, uint64_t scale);
__device__ void CUDAcombine(double* in1, unsigned __int64 cin1, double* in2, unsigned __int64 cin2);
__device__ void CUDAswap(double* in);

// Host function, will access on CPU
__host__ void CPUmergesort(double* in, unsigned __int64 count);
__host__ void CPUsplit(double* input, unsigned __int64 cin, double** out1, unsigned __int64* cout1, double** out2, unsigned __int64* cout2);
__host__ void CPUcombine(double* in1, unsigned __int64 cin1, double* in2, unsigned __int64 cin2);
__host__ void CPUswap(double* in);
__host__ double timeStr(LARGE_INTEGER t1, LARGE_INTEGER ts);
__host__ void __stdcall gChild(void** parg); // Child thread

// Cooperative function, start from CPU, will transport a rate of data to access on GPU
__host__ void COORPmergesort(double* in, unsigned __int64 count, double rate);

#endif