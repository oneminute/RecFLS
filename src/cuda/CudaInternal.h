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
        float minDepth;
        float maxDepth;
        int colorWidth;
        int colorHeight;
        int depthWidth;
        int depthHeight;
        float depthShift;
        int normalKernelHalfSize;
        float normalKernelMaxDistance;
        //int neighbourRadius;
        //float neighbourDistance;
        float boundaryGaussianSigma;
        int boundaryGaussianRadius;
        int boundaryEstimationRadius;
        float boundaryEstimationDistance;
        float boundaryAngleThreshold;
        int classifyRadius;
        float classifyDistance;
        int borderLeft;
        int borderRight;
        int borderTop;
        int borderBottom;
    };

    struct GpuFrame
    {
        pcl::gpu::DeviceArray2D<uchar3> colorImage;
        pcl::gpu::DeviceArray2D<ushort> depthImage;
        pcl::gpu::DeviceArray2D<int> indicesImage;
        pcl::gpu::DeviceArray<float3> pointCloud;
        pcl::gpu::DeviceArray<float3> pointCloudNormals;
        pcl::gpu::DeviceArray<float3> boundaryCloud;
        pcl::gpu::DeviceArray<uchar> boundaries;
        pcl::gpu::DeviceArray2D<uchar> boundaryImage;
        pcl::gpu::DeviceArray2D<int> boundaryIndices;

        Parameters parameters;

        bool allocate()
        {
            colorImage.create(parameters.colorHeight, parameters.colorWidth);
            depthImage.create(parameters.depthHeight, parameters.depthWidth);
            indicesImage.create(parameters.depthHeight, parameters.depthWidth);
            pointCloud.create(parameters.depthWidth * parameters.depthHeight);
            pointCloudNormals.create(parameters.depthWidth * parameters.depthHeight);
            boundaryCloud.create(parameters.depthWidth * parameters.depthHeight);
            boundaries.create(parameters.depthWidth * parameters.depthHeight);
            boundaryImage.create(parameters.depthHeight, parameters.depthWidth);
            boundaryIndices.create(parameters.depthHeight, parameters.depthWidth);
            return 1;
        }

        void free()
        {
            colorImage.release();
            depthImage.release();
            pointCloud.release();
            pointCloudNormals.release();
            boundaries.release();
            boundaryImage.release();
            boundaryIndices.release();
        }
    };

    void generatePointCloud(GpuFrame& frame);

    __global__ void eig(const float* M, float* V, float* L, const int n, bool useIterative);
    __global__ void eigVal(const float* M, float* L, const int n);
}

#endif // CUDAINTERNAL_H
