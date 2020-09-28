#pragma once
#include<cuda_runtime.h>

__host__ void CUDAmergesort(double* in, __int64 count);
__global__ void CM_slicer();