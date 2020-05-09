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
#include "FusedLineInternal.h"
#include "cuda.hpp"

namespace cuda
{
    __device__ __forceinline__ void covEig2(Mat22* m, float2* eigValue, Mat22* eigVec)
    {
        float a = m->rows[0].x;
        float b = m->rows[0].y;
        float c = m->rows[1].x;
        float d = m->rows[1].y;

        float a_d = a + d;
        float delta = sqrtf(a_d * a_d - 4 * (a * d - b * c));
        float gamma1 = (a_d - delta) / 2;
        float gamma2 = (a_d + delta) / 2;

        bool inverse = false;
        if (gamma1 < gamma2)
        {
            inverse = true;
        }

        float2 eigVec1;
        float2 eigVec2;
        eigVec1.x = 1;
        eigVec2.x = 1;
        if (a == 0)
        {
            eigVec1 = { 1, 0 };
            eigVec2 = { 0, 1 };
        }
        else if (d == 0)
        {
            eigVec1 = { 0, 1 };
            eigVec2 = { 1, 0 };
        }
        else if (b == 0)
        {
            if (a >= d)
            {
                eigVec1 = { 0, 1 };
                eigVec2 = { 1, 0 };
            }
            else
            {
                eigVec1 = { 1, 0 };
                eigVec2 = { 0, 1 };
            }
            //eigVec1.y = c / (gamma1 - d);
            //eigVec2.y = c / (gamma2 - d);
        }
        else
        {
            eigVec1.y = (gamma1 - a) / b;
            eigVec2.y = (gamma2 - a) / b;
        }

        eigVec1 = normalize(eigVec1);
        eigVec2 = normalize(eigVec2);
        
        if (inverse)
        {
            eigValue->x = gamma2;
            eigValue->y = gamma1;
            eigVec->rows[0].x = eigVec2.x;
            eigVec->rows[1].x = eigVec2.y;
            eigVec->rows[0].y = eigVec1.x;
            eigVec->rows[1].y = eigVec1.y;
        }
        else
        {
            eigValue->x = gamma1;
            eigValue->y = gamma2;
            eigVec->rows[0].x = eigVec1.x;
            eigVec->rows[1].x = eigVec1.y;
            eigVec->rows[0].y = eigVec2.x;
            eigVec->rows[1].y = eigVec2.y;
        }
        /*eigValue->x = gamma1;
            eigValue->y = gamma2;
            eigVec->rows[0].x = eigVec1.x;
            eigVec->rows[1].x = eigVec1.y;
            eigVec->rows[0].y = eigVec2.x;
            eigVec->rows[1].y = eigVec2.y;*/
    }

    struct EDLineExtractor
    {
        FusedLineParameters parameters;
        pcl::gpu::PtrStepSz<uchar> grayImage;
        pcl::gpu::PtrStepSz<ushort> gradientImage;
        pcl::gpu::PtrStepSz<uchar> dirImage;
        pcl::gpu::PtrStepSz<uchar> anchorImage;
        pcl::gpu::PtrStepSz<float> angleImage;
        pcl::gpu::PtrStepSz<float> radiusImage;
        pcl::gpu::PtrSz<float3> projCloud;

        __device__ __forceinline__ void sobelFilter()
        {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            int ix = index % parameters.rgbWidth;
            int iy = index / parameters.rgbWidth;

            if (ix == 0 || ix == parameters.rgbWidth - 1 || iy == 0 || iy == parameters.rgbHeight - 1)
                gradientImage[index] = parameters.gradientThreshold - 1;

            int bottomRightIndex = (iy + 1) * parameters.rgbWidth + (ix + 1);
            int topRightIndex = (iy - 1) * parameters.rgbWidth + (ix + 1);
            int bottomLeftIndex = (iy + 1) * parameters.rgbWidth + (ix - 1);
            int topLeftIndex = (iy - 1) * parameters.rgbWidth + (ix - 1);

            int rightIndex = iy * parameters.rgbWidth + ix + 1;
            int leftIndex = iy * parameters.rgbWidth + ix - 1;
            int topIndex = (iy - 1) * parameters.rgbWidth + ix;
            int bottomIndex = (iy + 1) * parameters.rgbWidth + ix;

            uchar bottomRightValue = grayImage[bottomRightIndex];
            uchar topRightValue = grayImage[topRightIndex];
            uchar bottomLeftValue = grayImage[bottomLeftIndex];
            uchar topLeftValue = grayImage[topLeftIndex];

            uchar rightValue = grayImage[rightIndex];
            uchar leftValue = grayImage[leftIndex];
            uchar topValue = grayImage[topIndex];
            uchar bottomValue = grayImage[bottomIndex];

            int com1 = bottomRightValue - topLeftValue;
            int com2 = topRightValue - bottomLeftValue;

            int gx = abs(com1 + com2 + 2 * (rightValue - leftValue));
            int gy = abs(com1 - com2 + 2 * (bottomValue - topValue));
            int sum = gx + gy;

            if (sum >= parameters.gradientThreshold)
            {
                if (gx >= gy)
                {
                    dirImage[index] = EDGE_VERTICAL;
                }
                else
                {
                    dirImage[index] = EDGE_HORIZONTAL;
                }
            }

            gradientImage[index] = sum;
        }

        __device__ __forceinline__ void extractAnchors()
        {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            int ix = index % parameters.rgbWidth;
            int iy = index / parameters.rgbWidth;
            
            /*if (index == 0)
            {
                printf("gradient threshold = %d\n", parameters.gradientThreshold);
            }*/
            if (ix <= 1 || ix >= parameters.rgbWidth - 2 || iy <= 1 || iy >= parameters.rgbHeight - 2)
                return;

            int curr = gradientImage[index];
            int left = gradientImage[index - 1];
            int right = gradientImage[index + 1];
            int top = gradientImage[(iy - 1) * parameters.rgbWidth + ix];
            int bottom = gradientImage[(iy + 1) * parameters.rgbWidth + ix];
            ushort dir = dirImage[index];
            
            if (dir == 0)
                return;

            int diff1 = -1;
            int diff2 = -1;
            if (dir == EDGE_VERTICAL)
            {
                diff1 = curr - left;
                diff2 = curr - right;
            }
            else if (dir = EDGE_HORIZONTAL)
            {
                diff1 = curr - top;
                diff2 = curr - bottom;
            }

            if (diff1 >= 0 && diff2 >= 0)
            {
                //printf("%d, %d: %d, %d, %d\n", ix, iy, diff1, diff2, curr);
                anchorImage[index] = ANCHOR_PIXEL;
            }
        }

        __device__ __forceinline__ void project()
        {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            int ix = index % parameters.rgbWidth;
            int iy = index / parameters.rgbWidth;
            float2 coord = { ix, iy };
            float2 primaDir = { 1, -1 };

            if (ix <= parameters.searchRadius || ix >= parameters.rgbWidth - parameters.searchRadius || iy <= parameters.searchRadius || iy >= parameters.rgbHeight - parameters.searchRadius)
                return;
            
            int dir = dirImage[index];
            int anchor = anchorImage[index];
            if (anchor != ANCHOR_PIXEL)
                return;

            //printf("%d\n", (int)index);

            bool valid = true;
            int it = 0;
            float2 neighbours[9 * 9] = { 0 };
            //bool done[9][9] = { false };
            //done[5][5] = true;
            //int2 localCenter = { 5, 5 };

            for (it = 0; it < 1; it++)
            {
                Mat22 covMat = makeMat22(0.f);
                float2 avg = { 0, 0 };
                int count = 0;
                for (int i = -4; i <= 4; i++)
                {
                    int absI = abs(i);
                    for (int j = -absI; j <= absI; j++)
                        //for (int j = -4; j <= 4; j++)
                    {
                        int nx = ix;
                        int ny = iy;
                        if (dir == EDGE_VERTICAL)
                        {
                            ny += i;
                            nx += j;
                        }
                        else if (dir == EDGE_HORIZONTAL)
                        {
                            nx += i;
                            ny += j;
                        }
                        
                        int nIndex = ny * parameters.rgbWidth + nx;
                        int nAnchor = anchorImage[nIndex];
                        if (nAnchor == ANCHOR_PIXEL)
                        {
                            float2 nCoord = { nx, ny };
                            float2 ray = coord - nCoord;
                            float fDist = sqrt(dot(ray, ray));
                            int iDist = round(fDist);
                            bool brk = false;
                            float2 cursor = ray;
                            float2 rayDir = normalize(ray);
                            for (int j = 1; j < iDist; j++)
                            {
                                float2 curr = cursor + rayDir * j;

                                int nnIndex = curr.y * parameters.rgbWidth + curr.x;
                                int nnAnchor = anchorImage[nnIndex];
                                /*if (ix == 296 && iy == 42)
                                {
                                    printf("  [%d %d]: %d [%d %d] %f %d\n", (int)cursor.x, (int)cursor.y, j, (int)curr.x, (int)curr.y, fDist, iDist);
                                }*/
                                if (nnAnchor != ANCHOR_PIXEL)
                                {
                                    brk = true;
                                    break;
                                }
                            }
                            if (!brk)
                            {
                                neighbours[count].x = nx;
                                neighbours[count].y = ny;
                                avg += neighbours[count];
                                count++;
                            }
                        }
                    }
                }
                
                avg /= count;
                if (count < 5)
                {
                    //printf("iteration = %d, count = %d, avg = [%f, %f]\n", it, count, avg.x, avg.y);
                    valid = false;
                    break;
                }

                int newCount = 0;
                for (int i = 0; i < count; i++)
                {
                    float2 diff = neighbours[i] - avg;
                    Mat22 m;
                    makeMat22(diff, diff, m);
                    covMat += m;

                    if (ix == 296 && iy == 42)
                    {
                        printf("  %d %d %f %f\n", (int)neighbours[i].x, (int)neighbours[i].y, diff.x, diff.y);
                    }
                }
                if (ix == 296 && iy == 42)
                    printf("new count: %d\n", newCount);

                covMat /= count;

                float2 eigValue;
                Mat22 eigVector;
                covEig2(&covMat, &eigValue, &eigVector);
                float2 dir;
                dir.x = eigVector.rows[0].x;
                dir.y = eigVector.rows[1].x;
                //if (dot(dir, primaDir) < 0)
                    //dir = -dir;
                //float angle = abs(qRadiansToDegrees(atan2f(dir.y, dir.x))) + 90;
                float angle = abs(atan2f(dir.y, dir.x)) * 2 / M_PI;
                float2 center = { parameters.rgbWidth / 2, parameters.rgbHeight / 2 };
                float2 ray = coord - center;
                float radius = cross(ray, dir);
                //float radius = sqrtf(dot(ray, ray));
                float2 projPoint = coord + dir * dot(ray, dir);
                projPoint.x /= parameters.rgbWidth;
                projPoint.y /= parameters.rgbHeight;

                projCloud[index].x = angle;
                projCloud[index].y = projPoint.x;
                projCloud[index].z = projPoint.y;

                angleImage[index] = angle;
                radiusImage[index] = radius / sqrtf(parameters.rgbWidth * parameters.rgbWidth / 4.f + parameters.rgbHeight * parameters.rgbHeight / 4.f);

                //printf("iteration = %d, count = %d, avg = [%f, %f], dir = [%f, %f], angle = %f, radius = %f\n", it, count, avg.x, avg.y, dir.x, dir.y, angle, radius);
                //if (index == 214200)
                //if (index == 17730)
                //if (index == 66994)
                //if (angle <= 0)
                //if (isnan(angle))
                /*if (ix > 433 && ix < 438 && iy >= 80 && iy <= 100)
                {
                    printf("[%d, %d]: iteration = %d, count = %d, avg = [%f, %f], dir = [%f, %f], angle = %f, radius = %f\n", ix, iy, it, count, avg.x, avg.y, dir.x, dir.y, angle, radius);

                }*/
                if (ix == 296 && iy == 42)
                {
                    printf("[%d, %d]: iteration = %d, count = %d, avg = [%f, %f], dir = [%f, %f], angle = %f, radius = %f\n", ix, iy, it, count, avg.x, avg.y, dir.x, dir.y, angle, radius);
                    printf("%d  covMat: %f %f\n", (int)index, covMat.rows[0].x, covMat.rows[0].y);
                    printf("%d          %f %f\n", (int)index, covMat.rows[1].x, covMat.rows[1].y);
                    printf("%d  eig value: %f %f\n", (int)index, eigValue.x, eigValue.y);
                    printf("%d  eig vector: %f %f\n", (int)index, eigVector.rows[0].x, eigVector.rows[0].y);
                    printf("%d              %f %f\n", (int)index, eigVector.rows[1].x, eigVector.rows[1].y);
                }
            }

            //if (it == 1 && !valid)
                //return;
        }

    };

    __global__ void sobelFilter(EDLineExtractor e)
    {
        e.sobelFilter();
    }

    __global__ void extractAnchors(EDLineExtractor e)
    {
        e.extractAnchors();
    }

    __global__ void project(EDLineExtractor e)
    {
        e.project();
    }

    void extractEDlines(FusedLineFrame& frame)
    {
        dim3 block(32);
        dim3 grid(frame.parameters.rgbWidth * frame.parameters.rgbHeight / block.x);

        int size = frame.parameters.rgbWidth * frame.parameters.rgbHeight;
        cudaMemset(frame.gradientImage.ptr(), 0, size * sizeof(short));
        cudaMemset(frame.dirImage.ptr(), 0, size);
        cudaMemset(frame.anchorImage.ptr(), 0, size);
        cudaMemset(frame.angleImage.ptr(), 0, size);
        cudaMemset(frame.radiusImage.ptr(), 0, size);
        cudaMemset(frame.projCloud.ptr(), 0, size * sizeof(float3));
        safeCall(cudaDeviceSynchronize());

        EDLineExtractor ee;
        ee.parameters = frame.parameters;
        ee.grayImage = frame.grayImage;
        ee.gradientImage = frame.gradientImage;
        ee.dirImage = frame.dirImage;
        ee.anchorImage = frame.anchorImage;
        ee.angleImage = frame.angleImage;
        ee.radiusImage = frame.radiusImage;
        ee.projCloud = frame.projCloud;

        TICK("cuda_sobel_filter");
        sobelFilter<<<grid, block>>>(ee);
        safeCall(cudaDeviceSynchronize());
        TOCK("cuda_sobel_filter");

        TICK("cuda_extract_anchors");
        extractAnchors<<<grid, block>>>(ee);
        safeCall(cudaDeviceSynchronize());
        TOCK("cuda_extract_anchors");

        TICK("cuda_project");
        project<<<grid, block>>>(ee);
        safeCall(cudaDeviceSynchronize());
        TOCK("cuda_project");
    }
}
