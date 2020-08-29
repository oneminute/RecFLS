#include "cuda_runtime.h"
#include "cuda.h"
#include <Eigen/Core>
#include <Eigen/Dense>

__device__ float3 d_mean(const float3* pts, const int n);
__device__ void d_cov(const float3* pts, float* C, const int n);
__global__ void mean(const float* pts, float* m, const int n);
__global__ void cov(const float* pts, float* C, const int n);

