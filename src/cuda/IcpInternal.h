#ifndef ICPINTERNAL_H
#define ICPINTERNAL_H

#include "cuda.h"

namespace cuda
{
    struct Mat33
    {
        float3 data[3];
    };

    struct IcpParameters
    {
        float cx;
        float cy;
        float fx;
        float fy;
        float minDepth;
        float maxDepth;
        int depthWidth;
        int depthHeight;
        float depthShift;
        float normalKnnRadius;
        int normalKernalRadius;
        float icpDistThreshold;
        float icpAnglesThreshold;
    };

    struct IcpFrame
    {
        pcl::gpu::DeviceArray2D<ushort> depthImage;
        pcl::gpu::DeviceArray<float3> pointCloud;
        pcl::gpu::DeviceArray<float3> pointCloudCache;
        pcl::gpu::DeviceArray<float3> pointCloudNormals;

        IcpParameters parameters;

        bool allocate()
        {
            depthImage.create(parameters.depthHeight, parameters.depthWidth);
            pointCloud.create(parameters.depthWidth * parameters.depthHeight);
            pointCloudCache.create(parameters.depthWidth * parameters.depthHeight);
            pointCloudNormals.create(parameters.depthWidth * parameters.depthHeight);
            return 1;
        }

        void free()
        {
            depthImage.release();
            pointCloud.release();
            pointCloudCache.release();
            pointCloudNormals.release();
        }

        void generateCloud(IcpFrame& frame1);
        void icp(IcpFrame& frame1, IcpFrame& frame2, const Mat33& initRot, const float3& initTrans, Mat33& rot, float3& trans, float& error);
    };
}

#endif // ICPINTERNAL_H