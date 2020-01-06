#ifndef CUDAINTERNAL_H
#define CUDAINTERNAL_H

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

namespace cuda
{
    struct Parameters
    {
        float cx;
        float cy;
        float fx;
        float fy;
        int colorWidth;
        int colorHeight;
        int depthWidth;
        int depthHeight;
        float depthShift;
    };

    struct Frame
    {
        pcl::gpu::DeviceArray2D<uchar3> colorImage;
        pcl::gpu::DeviceArray2D<ushort> depthImage;
        pcl::gpu::DeviceArray<float4> pointCloud;
        float* score;
    };

    void generatePointCloud(Parameters& parameters, Frame& frame);
}

#endif // CUDAINTERNAL_H
