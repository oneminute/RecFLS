#include "CudaInternal.h"

#include <opencv2/opencv.hpp>
#include <pcl/gpu/containers/device_array.h>

#include <cuda_runtime.h>

namespace cuda
{
    struct ExtractPointCloud
    {
        Parameters parameters;
        Frame frame;
        pcl::gpu::PtrSz<float4> points;
        pcl::gpu::PtrStepSz<uchar3> colorImage;
        pcl::gpu::PtrStepSz<ushort> depthImage;
        //mutable float4* points;

        __device__ __forceinline__ void operator() ()
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            //int iz = blockIdx.z * blockDim.z + threadIdx.z;

            //printf("ix: %d, iy: %d, iz: %d, threadIdx: (%d, %d, %d) blockIdx: (%d, %d, %d) blockDim: (%d, %d, %d) gridDim: (%d, %d, %d)\n",
               //ix, iy, iz,
               //threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z,
               //blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);

            if (index == 0)
            {
                printf("cx %f, cy %f, fx %f, fy %f, shift %f, width %d, height %d\n", parameters.cx, parameters.cy, parameters.fx, parameters.fy, parameters.depthShift, parameters.depthWidth, parameters.depthHeight);
            }

            int ix = index % parameters.depthWidth;
            int iy = index / parameters.depthWidth;

            float x = 0, y = 0, z = 0;
            float zValue = depthImage[index] * 1.f;
            z = zValue / parameters.depthShift;
            x = (ix - parameters.cx) * z / parameters.fx;
            y = (iy - parameters.cy) * z / parameters.fy;

            //pt.b = colorMat.at<cv::Vec3b>(cv::Point(j, i))[0];
            //pt.g = colorMat.at<cv::Vec3b>(cv::Point(j, i))[1];
            //pt.r = colorMat.at<cv::Vec3b>(cv::Point(j, i))[2];
            const float qnan = std::numeric_limits<float>::quiet_NaN ();

//            if(zValue > 0 && !qIsNaN(zValue)) {
            /*if (x > 0.1f && y > 0.1f && z > 0.1f) {
            }
            else
            {
                x = qnan;
                y = qnan;
                z = qnan;
            }*/

            points[index].x = x;
            if (index % 1024 == 0)
            {
                printf("index: %d, ix: %d, iy: %d, x: %f, y: %f, z: %f, depth: %d\n", index, ix, iy, x, y, z, depthImage[index]);
            }

            points[index].y = y;
            points[index].z = z;

            //points[index].x = x;
            //points[index].y = y;
            //points[index].z = z;
        }
    };

    __global__ void generatePointCloudKernal(ExtractPointCloud epc)
    {
        epc();
    }

    void generatePointCloud(Parameters& parameters, Frame& frame)
    {
        dim3 grid(parameters.depthWidth * parameters.depthHeight / 32);
        dim3 block(32);

        ExtractPointCloud epc;
        epc.parameters = parameters;
        epc.frame = frame;
        //epc.points = frame.pointCloud.ptr();
        epc.points = frame.pointCloud;
        epc.colorImage = frame.colorImage;
        epc.depthImage = frame.depthImage;

        generatePointCloudKernal<<<grid, block>>>(epc);
        safeCall(cudaDeviceSynchronize());
    }
}