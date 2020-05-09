#ifndef CUDAINTERNAL_H
#define CUDAINTERNAL_H

#include "cuda.h"

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
        float normalKnnRadius;
        int normalKernalRadius;
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
        int debugX;
        int debugY;
        int peakClusterTolerance;
        int minClusterPeaks;
        int maxClusterPeaks;
        float cornerHistSigma;
    };

    struct GpuFrame
    {
        pcl::gpu::DeviceArray2D<uchar3> colorImage;
        pcl::gpu::DeviceArray2D<ushort> depthImage;
        pcl::gpu::DeviceArray2D<int> indicesImage;
        pcl::gpu::DeviceArray<float3> pointCloud;
        pcl::gpu::DeviceArray<float3> pointCloudCache;
        pcl::gpu::DeviceArray<float3> pointCloudNormals;
        pcl::gpu::DeviceArray<uchar> boundaries;
        pcl::gpu::DeviceArray2D<uchar> boundaryImage;
        pcl::gpu::DeviceArray<uint> neighbours;

        Parameters parameters;

        bool allocate()
        {
            colorImage.create(parameters.colorHeight, parameters.colorWidth);
            depthImage.create(parameters.depthHeight, parameters.depthWidth);
            indicesImage.create(parameters.depthHeight, parameters.depthWidth);
            pointCloud.create(parameters.depthWidth * parameters.depthHeight);
            pointCloudCache.create(parameters.depthWidth * parameters.depthHeight);
            pointCloudNormals.create(parameters.depthWidth * parameters.depthHeight);
            boundaries.create(parameters.depthWidth * parameters.depthHeight);
            boundaryImage.create(parameters.depthHeight, parameters.depthWidth);
            return 1;
        }

        void free()
        {
            colorImage.release();
            depthImage.release();
            indicesImage.release();
            pointCloud.release();
            pointCloudCache.release();
            pointCloudNormals.release();
            boundaries.release();
            boundaryImage.release();
        }
    };

    void generatePointCloud(GpuFrame& frame);

    __global__ void eig(const float* M, float* V, float* L, const int n, bool useIterative);
    __global__ void eigVal(const float* M, float* L, const int n);
}

#endif // CUDAINTERNAL_H
