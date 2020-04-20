#ifndef CUDA_H
#define CUDA_H

#include <pcl/gpu/containers/device_array.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>

#define safeCall(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) \
    { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(-10 * error); \
    } \
}

#endif // CUDA_H