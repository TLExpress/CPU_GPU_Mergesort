#pragma once
#ifndef _MERGESORT_H
#define _MERGESORT_H
#include <cuda_runtime.h>
#include <helper_cuda.h>

__global__ void CUDAmergesort(double* in, __int64 count);
__host__ void CPUmergesort(double* in, __int64 count);
__host__ void COORPmergesort(double* in, __int64 count, double rate);
#endif