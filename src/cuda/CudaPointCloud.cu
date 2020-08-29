#include "CudaInternal.h"
#include "cutil_math2.h"
#include "cov.h"

#include <QtMath>
#include <opencv2/opencv.hpp>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/cuda/common/eigen.h>
#include <math.h>
#include <float.h>
#include "util/StopWatch.h"

namespace cuda
{
    struct ExtractPointCloud
    {
        Parameters parameters;
        pcl::gpu::PtrSz<float3> pointCloud;
        pcl::gpu::PtrSz<float3> pointCloudCache;
        pcl::gpu::PtrSz<float3> pointCloudNormals;
        pcl::gpu::PtrStepSz<int> indicesImage;
        //pcl::gpu::PtrStepSz<uchar3> colorImage;
        pcl::gpu::PtrStepSz<ushort> depthImage;
        pcl::gpu::PtrSz<uchar> boundaries;
        pcl::gpu::PtrStepSz<uchar> boundaryImage;
        pcl::gpu::PtrSz<uint> neighbours;

        __device__ bool isValid(float3 point)
        {
            if (isnan(point.x) || isnan(point.y) || isnan(point.z))
                return false;
            if (point.z < parameters.minDepth || point.z > parameters.maxDepth)
                return false;
            return true;
        }

        __device__ __forceinline__ void extracPointCloud()
        {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;

            if (index == 0)
            {
                printf("cx %f, cy %f, fx %f, fy %f, shift %f, width %d, height %d, boundary radius: %f, normalKernalRadius: %d, boundary angle threshold: %f, normal knn radius: %f, min cluster peaks: %d, max cluster peaks: %d\n", 
                    parameters.cx, parameters.cy, parameters.fx, parameters.fy, parameters.depthShift, parameters.depthWidth, parameters.depthHeight, 
                    parameters.boundaryEstimationDistance, parameters.normalKernalRadius, parameters.boundaryAngleThreshold, parameters.normalKnnRadius,
                    parameters.minClusterPeaks, parameters.maxClusterPeaks);
            }

            int ix = index % parameters.depthWidth;
            int iy = index / parameters.depthWidth;

            float x = 0, y = 0, z = 0;
            float zValue = depthImage[index] * 1.f;
            z = zValue / parameters.depthShift;
            x = (ix - parameters.cx) * z / parameters.fx;
            y = (iy - parameters.cy) * z / parameters.fy;

            float qnan = std::numeric_limits<float>::quiet_NaN();
            pointCloud[index].x = qnan;

            /*if (index % 1024 == 0)
            {
                printf("ix: %d, iy: %d, x: %f, y: %f, z: %f, depth: %d\n", ix, iy, x, y, z, depthImage[index]);
            }*/
            pointCloud[index].x = x;
            pointCloud[index].y = y;
            pointCloud[index].z = z;

            if (isValid(pointCloud[index]))
            {
                indicesImage[index] = index;
            }
        }

        __device__ __forceinline__ void estimateNormals()
        {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            int ix = index % parameters.depthWidth;
            int iy = index / parameters.depthWidth;

            float3 center = make_float3(0, 0, 0);
            float C_2d[3][3] = { {0,0,0},{0,0,0},{0,0,0} };
            float* C = (float*)C_2d; //rowwise
            float3 normal;
            int count = 0;

            float3 point = pointCloud[index];
            if (!isValid(point))
                return;

            float maxDistance = parameters.normalKnnRadius * parameters.normalKnnRadius;

            for (int j = max(0, iy - parameters.normalKernalRadius); j < min(iy + parameters.normalKernalRadius + 1, parameters.depthHeight); j++) {
                for (int i = max(0, ix - parameters.normalKernalRadius); i < min(ix + parameters.normalKernalRadius + 1, parameters.depthWidth); i++) {
                    size_t neighbourIndex = j * parameters.depthWidth + i;
                    float3 diff3 = pointCloud[neighbourIndex] - pointCloud[index];
                    float norm = dot(diff3, diff3);
                    //printf("index: %d, ix: %d, iy: %d, norm: %f, max distance: %f\n", index, ix, iy, norm, maxDistance);
                    if (norm < maxDistance) {
                        center += pointCloud[neighbourIndex];
                        outerAdd(pointCloud[neighbourIndex], C); //note: instead of pts[ind]-center, we demean afterwards
                        count++;
                    }
                }
            }
            //indices[0] = count;
            //printf("index: %d, ix: %d, iy: %d, x: %f, y: %f, z: %f, count: %d\n", index, ix, iy, normal.x, normal.y, normal.z, count);
            if (count < 3)
                return;

            center /= (count + 0.0f);
            outerAdd(center, C, -count);
            float fac = 1.0f / (count - 1.0f);
            mul(C, fac);
            float Q[3][3] = { {0,0,0},{0,0,0},{0,0,0} };
            float w[3] = { 0, 0, 0 };
            int result = dsyevv3(C_2d, Q, w);

            normal.x = Q[0][0];
            normal.y = Q[1][0];
            normal.z = Q[2][0];

            pointCloudNormals[index] = normalize(normal);

            //printf("index: %d, ix: %d, iy: %d, x: %f, y: %f, z: %f, depth: %d\n", index, ix, iy, normal.x, normal.y, normal.z, count);

            //.x = normal.x;
            //pointCloudNormals[index].y = normal.y;
            //pointCloudNormals[index].z = normal.z;
        }

        __device__ __forceinline__ void extractBoundaries()
        {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            int ix = index % parameters.depthWidth;
            int iy = index / parameters.depthWidth;

            int beAngles[360] = { 0 };
            float cornerAngles[360] = { 0 };
            float cornerValues[360] = { 0 };
            int count = 0;

            float3 point = pointCloud[index];
            if (!isValid(point))
                return;

            float maxDistance = parameters.boundaryEstimationDistance * parameters.boundaryEstimationDistance;

            float3 normal = pointCloudNormals[index];
            float3 beU, beV;
            beV = pcl::cuda::unitOrthogonal(normal);
            beU = cross(normal, beV);

            float3 cornerU, cornerV;
            cornerU.x = cornerU.y = cornerU.z = 0;
            float maxCornerAngle = 0;

            for (int j = max(0, iy - parameters.boundaryEstimationRadius); j < min(iy + parameters.boundaryEstimationRadius + 1, parameters.depthHeight); j++) {
                for (int i = max(0, ix - parameters.boundaryEstimationRadius); i < min(ix + parameters.boundaryEstimationRadius + 1, parameters.depthWidth); i++) {
                    size_t neighbourIndex = j * parameters.depthWidth + i;
                    float3 neighbourPoint = pointCloud[neighbourIndex];
                    float3 neighbourNormal = pointCloudNormals[neighbourIndex];
                    if (!isValid(neighbourPoint))
                        continue;

                    float3 diff3 = pointCloud[neighbourIndex] - pointCloud[index];
                    float norm = dot(diff3, diff3);
                    if (norm < maxDistance) {
                        float3 delta = neighbourPoint - point;

                        // calculate be angle.
                        float beAngle = atan2(dot(beV, delta), dot(beU, delta));
                        int beAngleIndex = (int)(beAngle * 180 / M_PI) + 180;
                        beAngles[beAngleIndex] += 1;

                        float cornerAngle = abs(acos(dot(neighbourNormal, normal)));
                        if (maxCornerAngle < cornerAngle)
                        {
                            maxCornerAngle = cornerAngle;
                            cornerU = neighbourNormal;
                        }

                        count++;
                    }
                }
            }
            cornerV = cross(cross(cornerU, normal), normal);
            cornerU = normal;

            for (int j = max(0, iy - parameters.boundaryEstimationRadius); j < min(iy + parameters.boundaryEstimationRadius + 1, parameters.depthHeight); j++) {
                for (int i = max(0, ix - parameters.boundaryEstimationRadius); i < min(ix + parameters.boundaryEstimationRadius + 1, parameters.depthWidth); i++) {
                    size_t neighbourIndex = j * parameters.depthWidth + i;
                    float3 neighbourPoint = pointCloud[neighbourIndex];
                    float3 neighbourNormal = pointCloudNormals[neighbourIndex];
                    if (!isValid(neighbourPoint))
                        continue;

                    float3 diff3 = pointCloud[neighbourIndex] - pointCloud[index];
                    float norm = dot(diff3, diff3);
                    if (norm < maxDistance) {
                        float3 delta = neighbourPoint - point;
                        float angle = atan2(dot(cornerV, neighbourNormal), dot(cornerU, neighbourNormal));
                        int angleIndex = (int)(angle * 180 / M_PI) + 180;
                        cornerAngles[angleIndex] += 1;

                        count++;
                    }
                }
            }

            {
                int angleDiff = 0;
                int maxDiff = 0;
                int lastAngle = 0;
                int firstAngle = lastAngle;
                for (int i = 1; i < 360; i++)
                {
                    if (beAngles[i] == 0)
                        continue;

                    if (firstAngle == 0)
                    {
                        firstAngle = i;
                    }

                    if (lastAngle == 0)
                    {
                        lastAngle = i;
                        continue;
                    }

                    angleDiff = i - lastAngle;
                    if (maxDiff < angleDiff)
                        maxDiff = angleDiff;

                    lastAngle = i;
                }

                angleDiff = 360 - lastAngle + firstAngle;
                if (maxDiff < angleDiff)
                    maxDiff = angleDiff;

                int pxX = (int)(point.x * parameters.fx / point.z + parameters.cx);
                int pxY = (int)(point.y * parameters.fy / point.z + parameters.cy);

                pxX = max(0, pxX);
                pxX = min(parameters.depthWidth - 1, pxX);
                pxY = max(0, pxY);
                pxY = min(parameters.depthHeight - 1, pxY);

                if (maxDiff > parameters.boundaryAngleThreshold)
                {
                    if (pxX <= parameters.borderLeft || pxY <= parameters.borderTop || pxX >= (parameters.depthWidth - parameters.borderRight) || pxY >= (parameters.depthHeight - parameters.borderBottom))
                    {
                        boundaries[index] = 1;
                        boundaryImage[pxY * parameters.depthWidth + pxX] = 1;
                    }
                    else
                    {
                        boundaries[index] = 3;
                        boundaryImage[pxY * parameters.depthWidth + pxX] = 3;
                    }
                }
            }
            {
                float maxValue;
                float avgValue = 0;
                count = 0;
                // 1-dimension filter
                //float kernel[5] = { -1, -1, 5, -1, -1 };
                for (int i = 0; i < 360; i++)
                {
                    if (cornerAngles[i] <= 0)
                        continue;

                    float value = cornerAngles[i] * cornerAngles[i];
                    if (maxValue < value)
                    {
                        maxValue = value;
                    }
                    cornerValues[i] = value;
                    avgValue += cornerValues[i];
                    count++;
                }
                avgValue /= count;

                int peaks = 0;
                int clusterPeaks = 0;
                float sigma = 0;
                int start = -1, end = -1;
                if (count == 1)
                {
                    maxValue = avgValue;
                    for (int i = 0; i < 360; i++)
                    {
                        if (cornerValues[i] > 0)
                        {
                            cornerValues[i] = cornerValues[i] / maxValue;
                            start = end = i;
                            break;
                        }
                    }
                    peaks = 1;
                }
                else
                {
                    count = 0;
                    avgValue = 0;
                    for (int i = 0; i < 360; i++)
                    {
                        if (cornerValues[i] <= 0)
                            continue;

                        cornerValues[i] = cornerValues[i] / maxValue;
                        avgValue += cornerValues[i];

                        if (cornerValues[i] > 0)
                        {
                            if (start == -1)
                            {
                                start = i;
                            }
                            end = i;
                        }
                        count++;
                    }
                    avgValue /= count;

                    count = 0;
                    for (int i = 0; i < 360; i++)
                    {
                        if (cornerValues[i] <= 0)
                            continue;

                        float diff = abs(cornerValues[i] - avgValue);
                        sigma += diff * diff;
                        count += 1;
                    }
                    sigma = sqrt(sigma / count);

                    float avgPeak = 0;
                    for (int i = 0; i < 360; i++)
                    {
                        float diff = abs(cornerValues[i] - avgValue);
                        cornerAngles[i] = -1;
                        if (cornerValues[i] > avgValue + sigma * parameters.cornerHistSigma)
                            //if (values[i] > avgValue)
                        {
                            cornerAngles[peaks] = i;
                            if (ix == parameters.debugX && iy == parameters.debugY)
                                printf("  %f %f %f %f\n", cornerAngles[i], cornerValues[i], diff, avgValue);
                            peaks++;
                        }
                    }

                    if (peaks > 0)
                        clusterPeaks = 1;

                    float avgClusterPeaks = 0;
                    for (int i = 0; i < peaks - 1; i++)
                    {
                        int index = static_cast<int>(cornerAngles[i]);
                        int nextIndex = static_cast<int>(cornerAngles[i + 1]);
                        cornerAngles[clusterPeaks - 1] += cornerValues[index];
                        if (cornerAngles[clusterPeaks - 1] > index)
                            cornerAngles[clusterPeaks - 1] -= index;
                        if (ix == parameters.debugX && iy == parameters.debugY)
                            printf("  %d %d %f %f\n", i, clusterPeaks, cornerAngles[clusterPeaks - 1], cornerValues[index]);
                        if (nextIndex - index > parameters.peakClusterTolerance)
                        {
                            //avgClusterPeaks += cornerAngles[clusterPeaks - 1];
                            clusterPeaks++;
                        }
                    }
                    for (int i = 0; i < clusterPeaks; i++)
                    {
                        avgClusterPeaks += cornerAngles[i];
                    }
                    avgClusterPeaks /= clusterPeaks;

                    count = 0;
                    for (int i = 0; i < clusterPeaks; i++)
                    {
                        if (ix == parameters.debugX && iy == parameters.debugY)
                            printf("  %d %f %f\n", i, cornerAngles[i], avgClusterPeaks);
                        if (cornerAngles[i] > avgClusterPeaks)
                            count++;
                        else if (abs(cornerAngles[i] - avgClusterPeaks) / cornerAngles[i] < 0.2f)
                            count++;
                    }
                    clusterPeaks = count;
                }

                /*if (ix == parameters.debugX && iy == parameters.debugY)
                {
                    int peakCount = 0;
                    for (int i = start; i <= end; i++)
                    {
                        bool isPeak = false;
                        if (i == cornerAngles[peakCount])
                        {
                            isPeak = true;
                            peakCount++;
                        }
                        printf("    %4d: %2.6f %d ", i, cornerValues[i], isPeak);
                        int pluses = ceil(cornerValues[i] / 0.05f);
                        for (int j = 0; j < pluses; j++)
                        {
                            printf("+");
                        }
                        printf("\n");
                    }
                    printf("ix: %4ld, iy: %4ld, count: %2d, sigma: %3.6f, peaks: %2d, cluster peaks: %2d, avgValue: %f, maxValue: %f\n",
                        ix, iy, count, sigma, peaks, clusterPeaks, avgValue, maxValue);
                }*/

                if (count > 1 && clusterPeaks >= parameters.minClusterPeaks && clusterPeaks <= parameters.maxClusterPeaks)
                {
                    if (!(ix <= parameters.borderLeft || iy <= parameters.borderTop || ix >= (parameters.depthWidth - parameters.borderRight) || iy >= (parameters.depthHeight - parameters.borderBottom)))
                    {
                        boundaries[index] = 4;
                        boundaryImage[index] = 4;
                    }
                }
            }
        }

        __device__ __forceinline__ void classifyBoundaries()
        {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            int ix = index % parameters.depthWidth;
            int iy = index / parameters.depthWidth;
            if (index == 0)
            {
                printf("classifyRadius: %d\n", parameters.classifyRadius);
            }

            if (boundaryImage[index] <= 1)
                return;

            float3 point = pointCloud[index];
            if (!isValid(point))
                return;

            if (boundaryImage[index] == 4)
            {
                bool valid = true;
                for (int a = max(iy - 15, 0); a <= min(iy + 15, parameters.depthHeight - 1); a++)
                {
                    for (int b = max(ix - 15, 0); b <= min(ix + 15, parameters.depthWidth - 1); b++)
                    {
                        int nPxIndex = a * parameters.depthWidth + b;
                        if (boundaries[nPxIndex] > 0 && boundaries[nPxIndex] < 4)
                        {
                            valid = false;
                            break;
                        }
                    }
                    if (!valid)
                    {
                        break;
                    }
                }
                if (valid)
                {
                    boundaries[index] = 4;
                    boundaryImage[index] = 4;
                }
                else
                {
                    boundaries[index] = 0;
                    boundaryImage[index] = 0;
                }
            }
            else
            {
                float2 original = { parameters.cx, parameters.cy };
                float2 coord = { static_cast<float>(ix), static_cast<float>(iy) };

                float2 ray = normalize(original - coord);
                //coord += -ray * 5;

                bool veil = false;
                for (int i = -parameters.classifyRadius; i < parameters.classifyRadius + 1; i++)
                {
                    float2 cursor = coord + ray * i;
                    int cursorX = floor(cursor.x);
                    int cursorY = floor(cursor.y);

                    for (int a = max(cursorY - 2, 0); a <= min(cursorY + 2, parameters.depthHeight - 1); a++)
                    {
                        for (int b = max(cursorX - 2, 0); b <= min(cursorX + 2, parameters.depthWidth - 1); b++)
                        {
                            int nPxIndex = a * parameters.depthWidth + b;
                            //int nPtIndex = static_cast<int>(nPxIndex);
                            if (boundaries[nPxIndex] <= 0 || boundaries[nPxIndex] == 4)
                                continue;

                            float3 nPt = pointCloud[nPxIndex];
                            if (!isValid(nPt))
                                continue;

                            float3 diff = point - nPt;
                            float dist = sqrt(dot(diff, diff));
                            if ((point.z - nPt.z) >= parameters.classifyDistance)
                            {
                                veil = true;
                                break;
                            }
                        }
                        if (veil)
                            break;
                    }
                    if (veil)
                        break;
                }
                if (veil)
                {
                    boundaries[index] = 2;
                    boundaryImage[index] = 2;
                }
            }
        }

        __device__ __forceinline__ void gaussianBlur()
        {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            int ix = index % parameters.depthWidth;
            int iy = index / parameters.depthWidth;

            float3 point = pointCloud[index];
            if (!isValid(point))
                return;

            int radius = parameters.boundaryGaussianRadius;
            float sigma = parameters.boundaryGaussianSigma;
            float value = 0;
            float sumG = 0;
            for (int j = max(0, iy - radius); j < min(iy + radius + 1, parameters.depthHeight); j++)
            {
                for (int i = max(0, ix - radius); i < min(ix + radius + 1, parameters.depthWidth); i++)
                {
                    size_t neighbourIndex = j * parameters.depthWidth + i;
                    float3 neighbourPoint = pointCloud[neighbourIndex];

                    if (!isValid(neighbourPoint))
                        continue;

                    float3 diff = point - neighbourPoint;
                    float dist = sqrt(dot(diff, diff));
                    if (dist < parameters.boundaryEstimationDistance)
                    {
                        float g = 1 / (2 * M_PI * sigma * sigma) * powf(M_E, -((i - ix) * (i - ix) + (j - iy) * (j - iy)) / (2 * sigma * sigma));
                        value += g * neighbourPoint.z;
                        //value += neighbourPoint.z;
                        sumG += g;
                        //sumG++;
                    }
                }
            }
            value /= sumG;
            if (isnan(value))
                return;

            //printf("index: %lld, ix: %d, iy: %d, z: %f, value: %f\n", index, ix, iy, pointCloud[index].z, value);

            pointCloudCache[index].x = point.x;
            pointCloudCache[index].y = point.y;
            pointCloudCache[index].z = value;
        }
    };

    __global__ void extractPointCloud(ExtractPointCloud epc)
    {
        epc.extracPointCloud();
    }

    __global__ void estimateNormals(ExtractPointCloud epc)
    {
        epc.estimateNormals();
    }

    __global__ void extractBoundaries(ExtractPointCloud epc)
    {
        epc.extractBoundaries();
    }

    __global__ void gaussianBlur(ExtractPointCloud epc)
    {
        epc.gaussianBlur();
    }

    __global__ void classifyBoundaries(ExtractPointCloud epc)
    {
        epc.classifyBoundaries();
    }

    void generatePointCloud(GpuFrame& frame)
    {
        dim3 block(16);
        dim3 grid(frame.parameters.depthWidth * frame.parameters.depthHeight / block.x);

        int size = frame.parameters.depthWidth * frame.parameters.depthHeight;
        cudaMemset(frame.pointCloud.ptr(), 0, size * sizeof(float3));
        cudaMemset(frame.pointCloudCache.ptr(), 0, size * sizeof(float3));
        cudaMemset(frame.pointCloudNormals.ptr(), 0, size * sizeof(float3));
        cudaMemset(frame.indicesImage.ptr(), -1, size * sizeof(int));
        cudaMemset(frame.boundaries.ptr(), 0, size * sizeof(uchar));
        cudaMemset(frame.boundaryImage.ptr(), 0, size * sizeof(uchar));
		cudaMemset(frame.pointsMat.ptr(), 0, size * sizeof(int));
        safeCall(cudaDeviceSynchronize());

        ExtractPointCloud epc;
        epc.parameters = frame.parameters;
        epc.pointCloud = frame.pointCloud;
        epc.pointCloudCache = frame.pointCloudCache;
        epc.pointCloudNormals = frame.pointCloudNormals;
        epc.indicesImage = frame.indicesImage;
        //epc.colorImage = frame.colorImage;
        epc.depthImage = frame.depthImage;
        epc.boundaries = frame.boundaries;
        epc.boundaryImage = frame.boundaryImage;
        epc.neighbours = frame.neighbours;
		

        TICK("cuda_extractPointCloud");
        extractPointCloud<<<grid, block>>>(epc);
        safeCall(cudaDeviceSynchronize());
        TOCK("cuda_extractPointCloud");

        //TICK("cuda_gaussianBlur");
        //gaussianBlur<<<grid, block>>>(epc);
        //safeCall(cudaDeviceSynchronize());
        //TOCK("cuda_gaussianBlur");

        //pcl::gpu::DeviceArray<float3> t = frame.pointCloud;
        //frame.pointCloudCache.copyTo(frame.pointCloud);
        //safeCall(cudaDeviceSynchronize());
        //cudaMemset(frame.pointCloudCache.ptr(), 0, size * sizeof(float3));
        //safeCall(cudaDeviceSynchronize());

        TICK("cuda_extimateNormals");
        estimateNormals<<<grid, block>>>(epc);
        safeCall(cudaDeviceSynchronize());
        TOCK("cuda_extimateNormals");

        //pcl::gpu::PtrSz<float3> t = epc.points;
        //epc.points = epc.pointsCache;
        //epc.pointsCache = t;

        TICK("cuda_extractBoundaries");
        extractBoundaries<<<grid, block>>>(epc);
        safeCall(cudaDeviceSynchronize());
        TOCK("cuda_extractBoundaries");

        //smoothBoudaries<<<grid, block>>>(epc);
        //safeCall(cudaDeviceSynchronize());

        TICK("cuda_classifyBoundaries");
        classifyBoundaries<<<grid, block>>>(epc);
        safeCall(cudaDeviceSynchronize());
        TOCK("cuda_classifyBoundaries");
    }

}