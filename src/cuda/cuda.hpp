#ifndef CUDA_HPP
#define CUDA_HPP

#include <cuda.h>
#include <pcl/cuda/cutil_math.h>
#include "cuda.h"

CUDA_FUNC_DECL float3 operator*(const Mat33& m, const float3& vec)
{
    return make_float3(dot(m.rows[0], vec), dot(m.rows[1], vec), dot(m.rows[2], vec));
}

CUDA_FUNC_DECL float3 toFloat3(const Eigen::Vector3f& v)
{
	float3 out;
	out.x = v.x();
	out.y = v.y();
	out.z = v.z();
	return out;
}

CUDA_FUNC_DECL bool isZero(const float3& v)
{
	return v.x == 0.f && v.y == 0.f && v.z == 0.f;
}

CUDA_FUNC_DECL bool isNan(const float3& v)
{
	return isnan(v.x) || isnan(v.y) || isnan(v.z);
}

CUDA_FUNC_DECL Mat33 makeMat33(float v)
{
	Mat33 m;
	for (int i = 0; i < 3; i++)
	{
		m.rows[i].x = v;
		m.rows[i].y = v;
		m.rows[i].z = v;
	}
	return m;
}

inline CUDA_FUNC_DECL Mat22 makeMat22(float v)
{
	Mat22 m;
	for (int i = 0; i < 2; i++)
	{
		m.rows[i].x = v;
		m.rows[i].y = v;
	}
	return m;
}

CUDA_FUNC_DECL void makeMat33(float3 a, float3 b, Mat33& m)
{
	m.rows[0].x = a.x * b.x;
	m.rows[0].y = a.x * b.y;
	m.rows[0].z = a.x * b.z;
	m.rows[1].x = a.y * b.x;
	m.rows[1].y = a.y * b.y;
	m.rows[1].z = a.y * b.z;
	m.rows[2].x = a.z * b.x;
	m.rows[2].y = a.z * b.y;
	m.rows[2].z = a.z * b.z;
}

inline CUDA_FUNC_DECL void makeMat22(float2 a, float2 b, Mat22& m)
{
	m.rows[0].x = a.x * b.x;
	m.rows[0].y = a.x * b.y;
	m.rows[1].x = a.y * b.x;
	m.rows[1].y = a.y * b.y;
}

CUDA_FUNC_DECL Mat33 makeMat33Identity()
{
	Mat33 m;
	makeMat33(0.f);
	m.rows[0].x = m.rows[1].y = m.rows[2].z = 1.f;
	return m;
}

CUDA_FUNC_DECL Mat33 toMat33(const Eigen::Matrix3f& m)
{
	Mat33 m33;
	m33.rows[0].x = m.row(0).x();
	m33.rows[0].y = m.row(0).y();
	m33.rows[0].z = m.row(0).z();
	m33.rows[1].x = m.row(1).x();
	m33.rows[1].y = m.row(1).y();
	m33.rows[1].z = m.row(1).z();
	m33.rows[2].x = m.row(2).x();
	m33.rows[2].y = m.row(2).y();
	m33.rows[2].z = m.row(2).z();
	return m33;
}

CUDA_FUNC_DECL Eigen::Matrix3f toMatrix3f(const Mat33& m33)
{
	Eigen::Matrix3f m3f;
	m3f.row(0).x() = m33.rows[0].x;
	m3f.row(0).y() = m33.rows[0].y;
	m3f.row(0).z() = m33.rows[0].z;
	m3f.row(1).x() = m33.rows[1].x;
	m3f.row(1).y() = m33.rows[1].y;
	m3f.row(1).z() = m33.rows[1].z;
	m3f.row(2).x() = m33.rows[2].x;
	m3f.row(2).y() = m33.rows[2].y;
	m3f.row(2).z() = m33.rows[2].z;
	return m3f;
}

CUDA_FUNC_DECL void operator+=(Mat33& a, Mat33& b)
{
	for (int i = 0; i < 3; i++)
	{
		a.rows[i] += b.rows[i];
	}
}

CUDA_FUNC_DECL void operator-=(Mat33& a, Mat33& b)
{
	for (int i = 0; i < 3; i++)
	{
		a.rows[i] -= b.rows[i];
	}
}

CUDA_FUNC_DECL void operator/=(Mat33& a, float b)
{
	for (int i = 0; i < 3; i++)
	{
		a.rows[i] /= b;
	}
}

inline CUDA_FUNC_DECL void operator+=(Mat22& a, Mat22& b)
{
	for (int i = 0; i < 2; i++)
	{
		a.rows[i] += b.rows[i];
	}
}

inline CUDA_FUNC_DECL void operator-=(Mat22& a, Mat22& b)
{
	for (int i = 0; i < 2; i++)
	{
		a.rows[i] -= b.rows[i];
	}
}

inline CUDA_FUNC_DECL void operator/=(Mat22& a, float b)
{
	for (int i = 0; i < 2; i++)
	{
		a.rows[i] /= b;
	}
}

CUDA_FUNC_DECL Eigen::Vector3f toVector3f(float3 a)
{
	Eigen::Vector3f v3f;
	v3f.x() = a.x;
	v3f.y() = a.y;
	v3f.z() = a.z;
	return v3f;
}

CUDA_FUNC_DECL bool isZero(const Mat33& m)
{
	return isZero(m.rows[0]) && isZero(m.rows[1]) && isZero(m.rows[2]);
}

CUDA_FUNC_DECL bool isNan(const Mat33& m)
{
	return isNan(m.rows[0]) || isNan(m.rows[1]) || isNan(m.rows[2]);
}

CUDA_FUNC_DECL float cross(float2 a, float2 b)
{
	return a.x * b.y - a.y * b.x;
}

#endif // CUDA_HPP