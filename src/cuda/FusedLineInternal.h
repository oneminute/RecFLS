#ifndef FUSEDLINEINTERNAL_H
#define FUSEDLINEINTERNAL_H

#include "cuda.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/cudafilters.hpp>

#define EDGE_VERTICAL 1
#define EDGE_HORIZONTAL 2
#define ANCHOR_PIXEL 254
#define EDGE_PIXEL 255

namespace cuda
{
    struct FusedLineParameters
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
        int rgbWidth;
        int rgbHeight;
        float normalKnnRadius;
        int normalKernalRadius;
        int blockSize;
        int gradientThreshold;
        int searchRadius;
    };

    struct FusedLineFrame
    {
        pcl::gpu::DeviceArray2D<uchar3> rgbImage;
        pcl::gpu::DeviceArray2D<uchar> grayImage;
        pcl::gpu::DeviceArray2D<ushort> depthImage;
        pcl::gpu::DeviceArray2D<short> gradientImage;
        pcl::gpu::DeviceArray2D<uchar> dirImage;
        pcl::gpu::DeviceArray2D<uchar> anchorImage;
        pcl::gpu::DeviceArray2D<float> angleImage;
        pcl::gpu::DeviceArray2D<float> radiusImage;
        pcl::gpu::DeviceArray<float3> pointCloud;
        pcl::gpu::DeviceArray<float3> pointCloudNormals;
        pcl::gpu::DeviceArray<float3> projCloud;
        cv::cuda::GpuMat rgbMatGpu;
        cv::cuda::GpuMat grayMatGpu;
        cv::cuda::GpuMat gradientMatGpu;
        cv::cuda::GpuMat depthMatGpu;
        cv::cuda::GpuMat anchorMatGpu;
        cv::cuda::GpuMat angleMatGpu;
        cv::cuda::GpuMat radiusMatGpu;

        FusedLineParameters parameters;

        bool allocate()
        {
            rgbImage.create(parameters.rgbHeight, parameters.rgbWidth);
            grayImage.create(parameters.rgbHeight, parameters.rgbWidth);
            depthImage.create(parameters.depthHeight, parameters.depthWidth);
            gradientImage.create(parameters.rgbHeight, parameters.rgbWidth);
            dirImage.create(parameters.rgbHeight, parameters.rgbWidth);
            anchorImage.create(parameters.rgbHeight, parameters.rgbWidth);
            angleImage.create(parameters.rgbHeight, parameters.rgbWidth);
            radiusImage.create(parameters.rgbHeight, parameters.rgbWidth);
            pointCloud.create(parameters.depthWidth * parameters.depthHeight);
            pointCloudNormals.create(parameters.depthWidth * parameters.depthHeight);
            projCloud.create(parameters.rgbHeight * parameters.rgbWidth);

            rgbMatGpu = cv::cuda::GpuMat(parameters.rgbHeight, parameters.rgbWidth, CV_8UC3, rgbImage);
            grayMatGpu = cv::cuda::GpuMat(parameters.rgbHeight, parameters.rgbWidth, CV_8UC1, grayImage);
            depthMatGpu = cv::cuda::GpuMat(parameters.depthHeight, parameters.depthWidth, CV_16U, depthImage);
            gradientMatGpu = cv::cuda::GpuMat(parameters.rgbHeight, parameters.rgbWidth, CV_16S, gradientImage);
            anchorMatGpu = cv::cuda::GpuMat(parameters.rgbHeight, parameters.rgbWidth, CV_8UC1, anchorImage);
            angleMatGpu = cv::cuda::GpuMat(parameters.rgbHeight, parameters.rgbWidth, CV_32F, angleImage);
            radiusMatGpu = cv::cuda::GpuMat(parameters.rgbHeight, parameters.rgbWidth, CV_32F, radiusImage);

            return 1;
        }

        void free()
        {
            rgbImage.release();
            grayImage.release();
            depthImage.release();
            gradientImage.release();
            dirImage.release();
            anchorImage.release();
            angleImage.release();
            radiusImage.release();
            pointCloud.release();
            pointCloudNormals.release();
            projCloud.release();
        }
    };

    void extractEDlines(FusedLineFrame& frame);
}

#endif