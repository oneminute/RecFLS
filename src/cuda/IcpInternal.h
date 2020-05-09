#ifndef ICPINTERNAL_H
#define ICPINTERNAL_H

#include "cuda.h"
#include <Eigen/Core>
#include <Eigen/Dense>

namespace cuda
{

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
        int icpKernalRadius;
        int blockSize;
    };

    struct IcpFrame
    {
        pcl::gpu::DeviceArray2D<ushort> depthImage;
        pcl::gpu::DeviceArray<float3> pointCloud;
        pcl::gpu::DeviceArray<float3> pointCloudNormals;

        IcpParameters parameters;

        bool allocate()
        {
            depthImage.create(parameters.depthHeight, parameters.depthWidth);
            pointCloud.create(parameters.depthWidth * parameters.depthHeight);
            pointCloudNormals.create(parameters.depthWidth * parameters.depthHeight);
            return 1;
        }

        void free()
        {
            depthImage.release();
            pointCloud.release();
            pointCloudNormals.release();
        }

    };

    struct IcpCache
    {
        IcpParameters parameters;
        pcl::gpu::DeviceArray<float3> srcCloud;
        pcl::gpu::DeviceArray<float3> dstCloud;
        //pcl::gpu::DeviceArray<Eigen::Vector3f> srcCache;
        //pcl::gpu::DeviceArray<Eigen::Vector3f> dstCache;
        pcl::gpu::DeviceArray<float3> srcNormals;
        pcl::gpu::DeviceArray<float3> dstNormals;
        pcl::gpu::DeviceArray<int> pairs;
        pcl::gpu::DeviceArray<float3> srcSum;
        pcl::gpu::DeviceArray<float3> dstSum;
        pcl::gpu::DeviceArray<Mat33> covMatrix;
        pcl::gpu::DeviceArray<int> counts;
        //pcl::gpu::DeviceArray<Eigen::Matrix3f> covMatrixCache;

        bool allocate()
        {
            int size = parameters.depthWidth * parameters.depthHeight;
            pairs.create(size);
            srcSum.create(size / parameters.blockSize);
            dstSum.create(size / parameters.blockSize);
            //srcCache.create(size);
            //dstCache.create(size);
            //covMatrixCache.create(size);
            covMatrix.create(size / parameters.blockSize);
            counts.create(size / parameters.blockSize);
            return 1;
        }

        void free()
        {
            pairs.release();
            srcSum.release();
            dstSum.release();
            //srcCache.release();
            //dstCache.release();
            //covMatrixCache.release();
            covMatrix.release();
            counts.release();
        }
    };

    void icpGenerateCloud(IcpFrame& frame);
    void icp(IcpCache& cache, const Mat33& initRot, const float3& initTrans, Mat33& covMatrix, float3& avgSrc, float3& avgDst, float& error);
}

#endif // ICPINTERNAL_H