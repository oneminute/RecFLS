#include "cutil_math2.h"
#include "cov.h"

#include <QtMath>
#include <opencv2/opencv.hpp>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/cuda/common/eigen.h>
#include <math.h>
#include <float.h>
#include "util/StopWatch.h"

#include <cuda_runtime.h>
#include "IcpInternal.h"
#include "cuda.hpp"

namespace cuda
{

    struct ICP
    {
        IcpParameters parameters;
        pcl::gpu::PtrStepSz<ushort> srcDepthImage;
        pcl::gpu::PtrSz<float3> src;
        pcl::gpu::PtrSz<float3> dst;
        pcl::gpu::PtrSz<float3> srcNormals;
        pcl::gpu::PtrSz<float3> dstNormals;
        pcl::gpu::PtrSz<float3> srcSum;
        pcl::gpu::PtrSz<float3> dstSum;
        pcl::gpu::PtrSz<Mat33> covMatrix;
        pcl::gpu::PtrSz<float> errors;
        pcl::gpu::PtrSz<int> pairs;
        pcl::gpu::PtrSz<int> counts;
        Mat33 initRot;
        float3 initTrans;

        __device__ bool isValid(float3& point)
        {
            if (isnan(point.x) || isnan(point.y) || isnan(point.z))
                return false;
            if (point.z < parameters.minDepth || point.z > parameters.maxDepth)
                return false;
            return true;
        }

        __device__ bool isValidCoord(int x, int y)
        {
            return x >= 0 && x < parameters.depthWidth&& y >= 0 && y < parameters.depthHeight;
        }

        __device__ __forceinline__ void extracPointCloud()
        {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            
            int ix = index % parameters.depthWidth;
            int iy = index / parameters.depthWidth;

            float3 pt;
            float zValue = srcDepthImage[index] * 1.f;
            pt.z = zValue / parameters.depthShift;
            pt.x = (ix - parameters.cx) * pt.z / parameters.fx;
            pt.y = (iy - parameters.cy) * pt.z / parameters.fy;

            if (isValid(pt))
            {
                src[index] = pt;
            }
            else
            {
                src[index].x = src[index].y = src[index].z = std::numeric_limits<float>::quiet_NaN();
            }
        }

        __device__ __forceinline__ void estimateNormals()
        {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            int ix = index % parameters.depthWidth;
            int iy = index / parameters.depthWidth;

            float3 center = make_float3(0.f);
            float C_2d[3][3] = { {0,0,0},{0,0,0},{0,0,0} };
            float* C = (float*)C_2d; //rowwise
            float3 normal;
            int count = 0;

            float3 point;
            point = src[index];
            if (!isValid(src[index]))
                return;

            float maxDistance = parameters.normalKnnRadius * parameters.normalKnnRadius;

            for (int j = max(0, iy - parameters.normalKernalRadius); j < min(iy + parameters.normalKernalRadius + 1, parameters.depthHeight); j++) {
                for (int i = max(0, ix - parameters.normalKernalRadius); i < min(ix + parameters.normalKernalRadius + 1, parameters.depthWidth); i++) {
                    size_t neighbourIndex = j * parameters.depthWidth + i;
                    float3 diff3 = src[neighbourIndex] - src[index];
                    if (dot(diff3, diff3) < maxDistance) {
                        center += src[neighbourIndex];
                        outerAdd(src[neighbourIndex], C); //note: instead of pts[ind]-center, we demean afterwards
                        count++;
                    }
                }
            }
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

            //printf("index: %d, ix: %d, iy: %d, x: %f, y: %f, z: %f, depth: %d\n", index, ix, iy, normal.x, normal.y, normal.z, count);

            srcNormals[index].x = normal.x;
            srcNormals[index].y = normal.y;
            srcNormals[index].z = normal.z;
        }

        __device__ __forceinline__ void stepSearch()
        {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            __shared__ float3 blockSrcSum[32];
            __shared__ float3 blockDstSum[32];
            __shared__ bool valid[32];

            /*if (index == 0)
            {
                printf("%d %f %f\n", parameters.icpKernalRadius, parameters.icpDistThreshold, parameters.icpAnglesThreshold);
            }*/
            if (index == 0)
            {
                printf("cx %f, cy %f, fx %f, fy %f, shift %f, width %d, height %d, [%f, %f, %f, %f, %f, %f, %f, %f, %f] [%f, %f, %f]\n",
                    parameters.cx, parameters.cy, parameters.fx, parameters.fy, parameters.depthShift, parameters.depthWidth, parameters.depthHeight,
                    initRot.rows[0].x, initRot.rows[0].y, initRot.rows[0].z,
                    initRot.rows[1].x, initRot.rows[1].y, initRot.rows[1].z,
                    initRot.rows[2].x, initRot.rows[2].y, initRot.rows[2].z,
                    initTrans.x, initTrans.y, initTrans.z);
            }

            float3 ptSrc = src[index];
            float3 nmSrc = srcNormals[index];

            ptSrc = initRot * ptSrc + initTrans;
            nmSrc = initRot * nmSrc;

            int ix = ptSrc.x * parameters.fx / ptSrc.z + parameters.cx;
            int iy = ptSrc.y * parameters.fy / ptSrc.z + parameters.cy;

            float maxDistance = parameters.icpDistThreshold * parameters.icpDistThreshold;
            float maxValue = 10000;
            float3 ptDst;
            int dstIndex = -1;
            bool found = false;
            if (isValid(ptSrc) && ix >= 0 && iy >= 0 && ix < parameters.depthWidth && iy < parameters.depthHeight)
            {
                // 针对src点云中的每一个点，在dst中查找法线最佳匹配的近邻点。
                /*for (int j = max(0, iy - parameters.icpKernalRadius); j < min(iy + parameters.icpKernalRadius + 1, parameters.depthHeight); j++) 
                {
                    for (int i = max(0, ix - parameters.icpKernalRadius); i < min(ix + parameters.icpKernalRadius + 1, parameters.depthWidth); i++) 
                    {
                        int neighbourIndex = j * parameters.depthWidth + i;
                        if (!isValid(dst[neighbourIndex]))
                            continue;
                        float3 diff3 = dst[neighbourIndex] - ptSrc;
                        if (dot(diff3, diff3) < maxDistance) {
                            float cos = dot(nmSrc, dstNormals[neighbourIndex]);
                            if (cos < parameters.icpAnglesThreshold)
                                continue;

                            if (cos > maxValue)
                            {
                                maxValue = cos;
                                ptDst = dst[neighbourIndex];
                                dstIndex = neighbourIndex;
                                found = true;
                            }
                        }
                    }
                }*/

                for (int j = max(0, iy - parameters.icpKernalRadius); j < min(iy + parameters.icpKernalRadius + 1, parameters.depthHeight); j++) 
                {
                    for (int i = max(0, ix - parameters.icpKernalRadius); i < min(ix + parameters.icpKernalRadius + 1, parameters.depthWidth); i++) 
                    {
                        int neighbourIndex = j * parameters.depthWidth + i;
                        if (!isValid(dst[neighbourIndex]))
                            continue;
                        float3 diff3 = dst[neighbourIndex] - ptSrc;
                        float sqrDist = dot(diff3, diff3);
                        if (sqrDist < maxDistance) {

                            if (sqrDist < maxValue)
                            {
                                maxValue = sqrDist;
                                ptDst = dst[neighbourIndex];
                                dstIndex = neighbourIndex;
                                found = true;
                            }
                        }
                    }
                }
                
                /*dstIndex = iy * parameters.depthWidth + ix;
                if (isValid(dst[dstIndex]))
                {
                    float3 diff3 = dst[dstIndex] - ptSrc;
                    if (dot(diff3, diff3) < maxDistance) {
                        float cos = dot(nmSrc, dstNormals[dstIndex]);
                        if (cos >= parameters.icpAnglesThreshold)
                        {
                            ptDst = dst[dstIndex];
                            found = true;
                        }
                    }
                }*/
            }

            valid[threadIdx.x] = false;
            if (found)
            {
                blockSrcSum[threadIdx.x] = ptSrc;
                blockDstSum[threadIdx.x] = ptDst;
                pairs[index] = dstIndex;
                valid[threadIdx.x] = true;
            }
            int blockIndex = blockIdx.x * blockDim.x;
            __syncthreads();
            if (threadIdx.x == 0)
            {
                int blockMemIndex = blockIndex / parameters.blockSize;
                int count = 0;
                for (int i = 0; i < parameters.blockSize; i++)
                {
                    if (!valid[threadIdx.x])
                        continue;

                    srcSum[blockMemIndex] += blockSrcSum[i];
                    dstSum[blockMemIndex] += blockDstSum[i];
                    count++;
                }
                counts[blockMemIndex] = count;
            }
        }

        __device__ __forceinline__ void icpMatch(float3 avgSrc, float3 avgDst)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int blockIndex = blockIdx.x * blockDim.x;
            int blockMemIndex = blockIndex / parameters.blockSize;
            __shared__ Mat33 blockCovM[32];
            __shared__ int valid[32];

            int dstIndex = pairs[index];
            float3 ptSrc = initRot * src[index] + initTrans;
            float3 ptDst = make_float3(0.f);

            valid[threadIdx.x] = isValid(src[index]) && dstIndex >= 0;

            /*if (blockIndex >= 6775 && blockIndex <= 6785)
            {
                printf("%d %d %d %d [%f, %f, %f] [%f, %f, %f]\n", blockIndex, threadIdx.x, dstIndex, valid[threadIdx.x], ptSrc.x, ptSrc.y, ptSrc.z, ptDst.x, ptDst.y, ptDst.z);
            }*/

            if (valid[threadIdx.x])
            {
                ptDst = dst[dstIndex];
                ptSrc = ptSrc - avgSrc;
                ptDst = ptDst - avgDst;

                //printf("%d\n", threadIdx.x);
                makeMat33(ptSrc, ptDst, blockCovM[threadIdx.x]);
            }
            __syncthreads();

            ////printf("ix: %d, iy: %d, src: [%f, %f, %f], dst: [%f, %f, %f]\n", ix, iy, ptSrc.x(), ptSrc.y(), ptSrc.z(), ptDst.x(), ptDst.y(), ptDst.z());
            if (threadIdx.x == 0)
            {
                int count = 0;
                for (int i = 0; i < parameters.blockSize; i++)
                {
                    int isValid = valid[threadIdx.x];
                    if (isValid && !isNan(blockCovM[threadIdx.x]))
                    {
                        covMatrix[blockMemIndex] += blockCovM[i];
                        count++;
                    }
                }
                counts[blockMemIndex] = count;
            }
        }

        __device__ __forceinline__ void calculateErrors(Mat33 rot, float3 trans)
        {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            int blockIndex = blockIdx.x * blockDim.x;
            int blockMemIndex = blockIndex / parameters.blockSize;
            __shared__ float errorSmem[32];

            float3 ptSrc = rot * src[index] + trans;

            int ix = ptSrc.x * parameters.fx / ptSrc.z + parameters.cx;
            int iy = ptSrc.y * parameters.fy / ptSrc.z + parameters.cy;
            int dstIndex = iy * parameters.depthWidth + ix;
            float error = 0;
            if (dstIndex >= 0 && dstIndex < parameters.depthWidth * parameters.depthHeight)
            {
                float3 ptDst = dst[dstIndex];
                float3 nmDst = dstNormals[dstIndex];

                if (isValid(ptSrc) && isValid(ptDst) && isValidCoord(ix, iy))
                {
                    float3 dist = ptSrc - ptDst;
                    error = sqrtf(dot(dist, dist));
                }
            }
            errorSmem[threadIdx.x] = error;

            __syncthreads();
            if (threadIdx.x == 0)
            {
                errors[blockMemIndex] = 0;
                for (int i = 0; i < parameters.blockSize; i++)
                {
                    errors[blockMemIndex] += errorSmem[i];
                }
            }
        }
    };

    __global__ void extractPointCloud(ICP icp)
    {
        icp.extracPointCloud();
    }

    __global__ void estimateNormals(ICP icp)
    {
        icp.estimateNormals();
    }

    __global__ void stepSearch(ICP icp)
    {
        icp.stepSearch();
    }

    __global__ void icpMatch(ICP icp, float3 avgSrc, float3 avgDst)
    {
        icp.icpMatch(avgSrc, avgDst);
    }

    __global__ void calculateErrors(ICP icp, Mat33 rot, float3 trans)
    {
        icp.calculateErrors(rot, trans);
    }

    void icpGenerateCloud(IcpFrame& frame)
    {
        dim3 block(32);
        dim3 grid(frame.parameters.depthWidth * frame.parameters.depthHeight / block.x);

        int size = frame.parameters.depthWidth * frame.parameters.depthHeight;
        cudaMemset(frame.pointCloud.ptr(), 0, size * sizeof(float3));
        cudaMemset(frame.pointCloudNormals.ptr(), 0, size * sizeof(float3));
        safeCall(cudaDeviceSynchronize());

        ICP icp;
        icp.parameters = frame.parameters;
        icp.srcDepthImage = frame.depthImage;
        icp.src = frame.pointCloud;
        icp.srcNormals = frame.pointCloudNormals;

        TICK("cuda_extractPointCloud");
        extractPointCloud<<<grid, block>>>(icp);
        safeCall(cudaDeviceSynchronize());
        TOCK("cuda_extractPointCloud");

        TICK("cuda_estimateNormals");
        estimateNormals<<<grid, block>>>(icp);
        safeCall(cudaDeviceSynchronize());
        TOCK("cuda_estimateNormals");
    }

    void icp(IcpCache& cache, const Mat33& initRot, const float3& initTrans, Mat33& covMatrix, float3& avgSrc, float3& avgDst, int& pairsCout)
    {
        cache.parameters.blockSize = 32;

        dim3 block(cache.parameters.blockSize);
        dim3 grid(cache.parameters.depthWidth * cache.parameters.depthHeight / block.x);

        int size = cache.parameters.depthWidth * cache.parameters.depthHeight;
        cudaMemset(cache.pairs.ptr(), -1, size * sizeof(int));
        cudaMemset(cache.srcSum.ptr(), 0, size * sizeof(Eigen::Vector3f) / cache.parameters.blockSize);
        cudaMemset(cache.dstSum.ptr(), 0, size * sizeof(Eigen::Vector3f) / cache.parameters.blockSize);
        cudaMemset(cache.srcNormals.ptr(), 0, size * sizeof(Eigen::Vector3f) / cache.parameters.blockSize);
        cudaMemset(cache.dstNormals.ptr(), 0, size * sizeof(Eigen::Vector3f) / cache.parameters.blockSize);
        cudaMemset(cache.covMatrix.ptr(), 0, size * sizeof(Eigen::Matrix3f) / cache.parameters.blockSize);
        cudaMemset(cache.errors.ptr(), 0, size * sizeof(float) / cache.parameters.blockSize);
        safeCall(cudaDeviceSynchronize());

        ICP icp;
        icp.parameters = cache.parameters;
        icp.src = cache.srcCloud;
        icp.dst = cache.dstCloud;
        icp.srcNormals = cache.srcNormals;
        icp.dstNormals = cache.dstNormals;
        //icp.srcCache = cache.srcCache;
        //icp.dstCache = cache.dstCache;
        icp.covMatrix = cache.covMatrix;
        icp.errors = cache.errors;
        //icp.covMatrixCache = cache.covMatrixCache;
        icp.pairs = cache.pairs;
        icp.srcSum = cache.srcSum;
        icp.dstSum = cache.dstSum;
        icp.counts = cache.counts;
        icp.initRot = initRot;
        icp.initTrans = initTrans;

        TICK("cuda_icp_stepSearch");
        stepSearch<<<grid, block>>>(icp);
        safeCall(cudaDeviceSynchronize());
        TOCK("cuda_icp_stepSearch");

        std::vector<float3> srcSum(grid.x);
        cache.srcSum.download(srcSum);
        std::vector<float3> dstSum(grid.x);
        cache.dstSum.download(dstSum);
        std::vector<int> counts(grid.x);
        cache.counts.download(counts);

        avgSrc = make_float3(0.f);
        avgDst = make_float3(0.f);
        int count = 0;
        for (int i = 0; i < grid.x; i++)
        {
            int countI = counts[i];
            if (countI)
            {
                avgSrc += srcSum[i];
                avgDst += dstSum[i];
                count += countI;
            }
            //std::cout << i << ". [" << avgSrc.x << ", " << avgSrc.y << ", " << avgSrc.z << "] [" << avgDst.x << ", " << avgDst.y << ", " << avgDst.z << "] count = " << count << std::endl;
        }
        avgSrc /= count;
        avgDst /= count;

        std::cout << "avgSrc = [" << avgSrc.x << ", " << avgSrc.y << ", " << avgSrc.z << "], count = " << count << std::endl;
        std::cout << "avgDst = [" << avgDst.x << ", " << avgDst.y << ", " << avgDst.z << "], count = " << count << std::endl;

        TICK("cuda_icp_match");
        icpMatch<<<grid, block>>>(icp, avgSrc, avgDst);
        safeCall(cudaDeviceSynchronize());
        TOCK("cuda_icp_match");

        std::vector<Mat33> covMatrixes;
        cache.covMatrix.download(covMatrixes);
        cache.counts.download(counts);

        covMatrix = makeMat33(0.f);
        count = 0;
        for (int i = 0; i < grid.x; i++)
        {
            if (counts[i])
            {
                covMatrix += covMatrixes[i];
                count += counts[i];
                if (isNan(covMatrixes[i]))
                {
                    std::cout << i << std::endl;
                    std::cout << toMatrix3f(covMatrixes[i]) << std::endl;
                }
            }
        }
        covMatrix /= count;
        std::cout << "cov matrix count = " << count << std::endl;
    }

    void calculateErrors(IcpCache& cache, const Mat33& rot, const float3& trans, int pairsCount, float& error)
    {
        cache.parameters.blockSize = 32;

        dim3 block(cache.parameters.blockSize);
        dim3 grid(cache.parameters.depthWidth * cache.parameters.depthHeight / block.x);

        int size = cache.parameters.depthWidth * cache.parameters.depthHeight;
        cudaMemset(cache.errors.ptr(), 0, size * sizeof(float) / cache.parameters.blockSize);
        safeCall(cudaDeviceSynchronize());

        ICP icp;
        icp.parameters = cache.parameters;
        icp.src = cache.srcCloud;
        icp.dst = cache.dstCloud;
        icp.srcNormals = cache.srcNormals;
        icp.dstNormals = cache.dstNormals;
        icp.covMatrix = cache.covMatrix;
        icp.errors = cache.errors;

        TICK("cuda_calc_errors");
        calculateErrors << <grid, block >> > (icp, rot, trans);
        safeCall(cudaDeviceSynchronize());
        TOCK("cuda_calc_errors");

        std::vector<float> errors;
        cache.errors.download(errors);
        for (int i = 0; i < grid.x; i++)
        {
            error += errors[i];
        }
        error /= pairsCount;
        std::cout << "original error: " << error << std::endl;
    }
    
}