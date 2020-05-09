#ifndef CUDA_H
#define CUDA_H

#include <pcl/gpu/containers/device_array.h>
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <Eigen/Core>
#include <Eigen/Dense>

#define CUDA_FUNC_DECL inline __host__ __device__

struct alignas(32) Mat33
{
    float3 rows[3];
};

struct alignas(32) Mat22
{
    float2 rows[2];
};

CUDA_FUNC_DECL float3 operator*(const Mat33& m, const float3& vec);
CUDA_FUNC_DECL float3 toFloat3(const Eigen::Vector3f& v);
CUDA_FUNC_DECL bool isZero(const float3& v);
CUDA_FUNC_DECL bool isNan(const float3& v);
CUDA_FUNC_DECL Mat33 makeMat33(float v);
CUDA_FUNC_DECL Mat22 makeMat22(float v);
CUDA_FUNC_DECL void makeMat33(float3 a, float3 b, Mat33& m);
CUDA_FUNC_DECL void makeMat22(float2 a, float2 b, Mat22& m);
CUDA_FUNC_DECL Mat33 makeMat33Identity();
CUDA_FUNC_DECL Mat33 toMat33(const Eigen::Matrix3f& m);
CUDA_FUNC_DECL Eigen::Matrix3f toMatrix3f(const Mat33& m33);
CUDA_FUNC_DECL void operator+=(Mat33& a, Mat33& b);
CUDA_FUNC_DECL void operator-=(Mat33& a, Mat33& b);
CUDA_FUNC_DECL void operator/=(Mat33& a, float b);
CUDA_FUNC_DECL void operator+=(Mat22& a, Mat22& b);
CUDA_FUNC_DECL void operator-=(Mat22& a, Mat22& b);
CUDA_FUNC_DECL void operator/=(Mat22& a, float b);
CUDA_FUNC_DECL Eigen::Vector3f toVector3f(float3 a);
CUDA_FUNC_DECL bool isZero(const Mat33& m);
CUDA_FUNC_DECL bool isNan(const Mat33& m);
CUDA_FUNC_DECL float cross(float2 a, float2 b);

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

#endif // CUDA_H