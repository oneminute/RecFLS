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

#include <cuda_runtime.h>

namespace cuda
{
#define SQR(x)      ((x)*(x))                        // x^2 
#define SQR_ABS(x)  (SQR(cfloat(x)) + SQR(cimag(x)))  // |x|^2

#define M_SQRT3    1.73205080756887729352744634151   // sqrt(3)

    // calculates eigenvalues of 2x2 float symmetric matrix
    __device__ void dsyevc2(float A, float B, float C, float* rt1, float* rt2) {
        float sm = A + C;
        float df = A - C;
        float rt = sqrt(SQR(df) + 4.0 * B * B);
        float t;

        if (sm > 0.0)
        {
            *rt1 = 0.5 * (sm + rt);
            t = 1.0 / (*rt1);
            *rt2 = (A * t) * C - (B * t) * B;
        }
        else if (sm < 0.0)
        {
            *rt2 = 0.5 * (sm - rt);
            t = 1.0 / (*rt2);
            *rt1 = (A * t) * C - (B * t) * B;
        }
        else       // This case needs to be treated separately to avoid div by 0
        {
            *rt1 = 0.5 * rt;
            *rt2 = -0.5 * rt;
        }
    }


    // ----------------------------------------------------------------------------
    // Calculates the eigensystem of a float symmetric 2x2 matrix
    //    [ A  B ]
    //    [ B  C ]
    // in the form
    //    [ A  B ]  =  [ cs  -sn ] [ rt1   0  ] [  cs  sn ]
    //    [ B  C ]     [ sn   cs ] [  0   rt2 ] [ -sn  cs ]
    // where rt1 >= rt2. Note that this convention is different from the one used
    // in the LAPACK routine DLAEV2, where |rt1| >= |rt2|.
    // ----------------------------------------------------------------------------
    __device__ void dsyev2(float A, float B, float C, float* rt1, float* rt2, float* cs, float* sn)
    {
        float sm = A + C;
        float df = A - C;
        float rt = sqrt(SQR(df) + 4.0 * B * B);
        float t;

        if (sm > 0.0)
        {
            *rt1 = 0.5 * (sm + rt);
            t = 1.0 / (*rt1);
            *rt2 = (A * t) * C - (B * t) * B;
        }
        else if (sm < 0.0)
        {
            *rt2 = 0.5 * (sm - rt);
            t = 1.0 / (*rt2);
            *rt1 = (A * t) * C - (B * t) * B;
        }
        else       // This case needs to be treated separately to avoid div by 0
        {
            *rt1 = 0.5 * rt;
            *rt2 = -0.5 * rt;
        }

        // Calculate eigenvectors
        if (df > 0.0)
            *cs = df + rt;
        else
            *cs = df - rt;

        if (fabs(*cs) > 2.0 * fabs(B))
        {
            t = -2.0 * B / *cs;
            *sn = 1.0 / sqrt(1.0 + SQR(t));
            *cs = t * (*sn);
        }
        else if (fabs(B) == 0.0)
        {
            *cs = 1.0;
            *sn = 0.0;
        }
        else
        {
            t = -0.5 * (*cs) / B;
            *cs = 1.0 / sqrt(1.0 + SQR(t));
            *sn = t * (*cs);
        }

        if (df > 0.0)
        {
            t = *cs;
            *cs = -(*sn);
            *sn = t;
        }
    }



    // ----------------------------------------------------------------------------
    // Calculates the eigenvalues of a symmetric 3x3 matrix A using Cardano's
    // analytical algorithm.
    // Only the diagonal and upper triangular parts of A are accessed. The access
    // is read-only.
    // ----------------------------------------------------------------------------
    // Parameters:
    //   A: The symmetric input matrix
    //   w: Storage buffer for eigenvalues
    // ----------------------------------------------------------------------------
    // Return value:
    //   0: Success
    //  -1: Error
    // ----------------------------------------------------------------------------
    __device__ int dsyevc3(float A[3][3], float w[3])
        
    {
        float m, c1, c0;

        // Determine coefficients of characteristic poynomial. We write
        //       | a   d   f  |
        //  A =  | d*  b   e  |
        //       | f*  e*  c  |
        float de = A[0][1] * A[1][2];                                    // d * e
        float dd = SQR(A[0][1]);                                         // d^2
        float ee = SQR(A[1][2]);                                         // e^2
        float ff = SQR(A[0][2]);                                         // f^2
        m = A[0][0] + A[1][1] + A[2][2];
        c1 = (A[0][0] * A[1][1] + A[0][0] * A[2][2] + A[1][1] * A[2][2])        // a*b + a*c + b*c - d^2 - e^2 - f^2
            - (dd + ee + ff);
        c0 = A[2][2] * dd + A[0][0] * ee + A[1][1] * ff - A[0][0] * A[1][1] * A[2][2]
            - 2.0 * A[0][2] * de;                                     // c*d^2 + a*e^2 + b*f^2 - a*b*c - 2*f*d*e)

        float p, sqrt_p, q, c, s, phi;
        p = SQR(m) - 3.0 * c1;
        q = m * (p - (3.0 / 2.0) * c1) - (27.0 / 2.0) * c0;
        sqrt_p = sqrt(fabs(p));

        phi = 27.0 * (0.25 * SQR(c1) * (p - c1) + c0 * (q + 27.0 / 4.0 * c0));
        phi = (1.0 / 3.0) * atan2(sqrt(fabs(phi)), q);

        c = sqrt_p * cos(phi);
        s = (1.0 / M_SQRT3) * sqrt_p * sin(phi);

        w[0] = (1.0 / 3.0) * (m - c);
        w[1] = w[0] + s;
        w[2] = w[0] + c;
        w[0] -= s;

        return 0;
    }


    // ----------------------------------------------------------------------------
    __device__ int dsyevv3(float A[3][3], float Q[3][3], float w[3])
        // ----------------------------------------------------------------------------
        // Calculates the eigenvalues and normalized eigenvectors of a symmetric 3x3
        // matrix A using Cardano's method for the eigenvalues and an analytical
        // method based on vector cross products for the eigenvectors.
        // Only the diagonal and upper triangular parts of A need to contain meaningful
        // values. However, all of A may be used as temporary storage and may hence be
        // destroyed.
        // ----------------------------------------------------------------------------
        // Parameters:
        //   A: The symmetric input matrix
        //   Q: Storage buffer for eigenvectors
        //   w: Storage buffer for eigenvalues
        // ----------------------------------------------------------------------------
        // Return value:
        //   0: Success
        //  -1: Error
        // ----------------------------------------------------------------------------
        // Dependencies:
        //   dsyevc3()
        // ----------------------------------------------------------------------------
        // Version history:
        //   v1.1 (12 Mar 2012): Removed access to lower triangualr part of A
        //     (according to the documentation, only the upper triangular part needs
        //     to be filled)
        //   v1.0: First released version
        // ----------------------------------------------------------------------------
    {
#ifndef EVALS_ONLY
        float norm;          // Squared norm or inverse norm of current eigenvector
        float n0, n1;        // Norm of first and second columns of A
        float n0tmp, n1tmp;  // "Templates" for the calculation of n0/n1 - saves a few FLOPS
        float thresh;        // Small number used as threshold for floating point comparisons
        float error;         // Estimated maximum roundoff error in some steps
        float wmax;          // The eigenvalue of maximum modulus
        float f, t;          // Intermediate storage
        int i, j;             // Loop counters
#endif

  // Calculate eigenvalues
        dsyevc3(A, w);

#ifndef EVALS_ONLY
        wmax = fabs(w[0]);
        if ((t = fabs(w[1])) > wmax)
            wmax = t;
        if ((t = fabs(w[2])) > wmax)
            wmax = t;
        thresh = SQR(8.0 * DBL_EPSILON * wmax);

        // Prepare calculation of eigenvectors
        n0tmp = SQR(A[0][1]) + SQR(A[0][2]);
        n1tmp = SQR(A[0][1]) + SQR(A[1][2]);
        Q[0][1] = A[0][1] * A[1][2] - A[0][2] * A[1][1];
        Q[1][1] = A[0][2] * A[0][1] - A[1][2] * A[0][0];
        Q[2][1] = SQR(A[0][1]);

        // Calculate first eigenvector by the formula
        //   v[0] = (A - w[0]).e1 x (A - w[0]).e2
        A[0][0] -= w[0];
        A[1][1] -= w[0];
        Q[0][0] = Q[0][1] + A[0][2] * w[0];
        Q[1][0] = Q[1][1] + A[1][2] * w[0];
        Q[2][0] = A[0][0] * A[1][1] - Q[2][1];
        norm = SQR(Q[0][0]) + SQR(Q[1][0]) + SQR(Q[2][0]);
        n0 = n0tmp + SQR(A[0][0]);
        n1 = n1tmp + SQR(A[1][1]);
        error = n0 * n1;

        if (n0 <= thresh)         // If the first column is zero, then (1,0,0) is an eigenvector
        {
            Q[0][0] = 1.0;
            Q[1][0] = 0.0;
            Q[2][0] = 0.0;
        }
        else if (n1 <= thresh)    // If the second column is zero, then (0,1,0) is an eigenvector
        {
            Q[0][0] = 0.0;
            Q[1][0] = 1.0;
            Q[2][0] = 0.0;
        }
        else if (norm < SQR(64.0 * DBL_EPSILON) * error)
        {                         // If angle between A[0] and A[1] is too small, don't use
            t = SQR(A[0][1]);       // cross product, but calculate v ~ (1, -A0/A1, 0)
            f = -A[0][0] / A[0][1];
            if (SQR(A[1][1]) > t)
            {
                t = SQR(A[1][1]);
                f = -A[0][1] / A[1][1];
            }
            if (SQR(A[1][2]) > t)
                f = -A[0][2] / A[1][2];
            norm = 1.0 / sqrt(1 + SQR(f));
            Q[0][0] = norm;
            Q[1][0] = f * norm;
            Q[2][0] = 0.0;
        }
        else                      // This is the standard branch
        {
            norm = sqrt(1.0 / norm);
            for (j = 0; j < 3; j++)
                Q[j][0] = Q[j][0] * norm;
        }


        // Prepare calculation of second eigenvector
        t = w[0] - w[1];
        if (fabs(t) > 8.0 * DBL_EPSILON * wmax)
        {
            // For non-degenerate eigenvalue, calculate second eigenvector by the formula
            //   v[1] = (A - w[1]).e1 x (A - w[1]).e2
            A[0][0] += t;
            A[1][1] += t;
            Q[0][1] = Q[0][1] + A[0][2] * w[1];
            Q[1][1] = Q[1][1] + A[1][2] * w[1];
            Q[2][1] = A[0][0] * A[1][1] - Q[2][1];
            norm = SQR(Q[0][1]) + SQR(Q[1][1]) + SQR(Q[2][1]);
            n0 = n0tmp + SQR(A[0][0]);
            n1 = n1tmp + SQR(A[1][1]);
            error = n0 * n1;

            if (n0 <= thresh)       // If the first column is zero, then (1,0,0) is an eigenvector
            {
                Q[0][1] = 1.0;
                Q[1][1] = 0.0;
                Q[2][1] = 0.0;
            }
            else if (n1 <= thresh)  // If the second column is zero, then (0,1,0) is an eigenvector
            {
                Q[0][1] = 0.0;
                Q[1][1] = 1.0;
                Q[2][1] = 0.0;
            }
            else if (norm < SQR(64.0 * DBL_EPSILON) * error)
            {                       // If angle between A[0] and A[1] is too small, don't use
                t = SQR(A[0][1]);     // cross product, but calculate v ~ (1, -A0/A1, 0)
                f = -A[0][0] / A[0][1];
                if (SQR(A[1][1]) > t)
                {
                    t = SQR(A[1][1]);
                    f = -A[0][1] / A[1][1];
                }
                if (SQR(A[1][2]) > t)
                    f = -A[0][2] / A[1][2];
                norm = 1.0 / sqrt(1 + SQR(f));
                Q[0][1] = norm;
                Q[1][1] = f * norm;
                Q[2][1] = 0.0;
            }
            else
            {
                norm = sqrt(1.0 / norm);
                for (j = 0; j < 3; j++)
                    Q[j][1] = Q[j][1] * norm;
            }
        }
        else
        {
            // For degenerate eigenvalue, calculate second eigenvector according to
            //   v[1] = v[0] x (A - w[1]).e[i]
            //   
            // This would floatly get to complicated if we could not assume all of A to
            // contain meaningful values.
            A[1][0] = A[0][1];
            A[2][0] = A[0][2];
            A[2][1] = A[1][2];
            A[0][0] += w[0];
            A[1][1] += w[0];
            for (i = 0; i < 3; i++)
            {
                A[i][i] -= w[1];
                n0 = SQR(A[0][i]) + SQR(A[1][i]) + SQR(A[2][i]);
                if (n0 > thresh)
                {
                    Q[0][1] = Q[1][0] * A[2][i] - Q[2][0] * A[1][i];
                    Q[1][1] = Q[2][0] * A[0][i] - Q[0][0] * A[2][i];
                    Q[2][1] = Q[0][0] * A[1][i] - Q[1][0] * A[0][i];
                    norm = SQR(Q[0][1]) + SQR(Q[1][1]) + SQR(Q[2][1]);
                    if (norm > SQR(256.0 * DBL_EPSILON)* n0) // Accept cross product only if the angle between
                    {                                         // the two vectors was not too small
                        norm = sqrt(1.0 / norm);
                        for (j = 0; j < 3; j++)
                            Q[j][1] = Q[j][1] * norm;
                        break;
                    }
                }
            }

            if (i == 3)    // This means that any vector orthogonal to v[0] is an EV.
            {
                for (j = 0; j < 3; j++)
                    if (Q[j][0] != 0.0)                                   // Find nonzero element of v[0] ...
                    {                                                     // ... and swap it with the next one
                        norm = 1.0 / sqrt(SQR(Q[j][0]) + SQR(Q[(j + 1) % 3][0]));
                        Q[j][1] = Q[(j + 1) % 3][0] * norm;
                        Q[(j + 1) % 3][1] = -Q[j][0] * norm;
                        Q[(j + 2) % 3][1] = 0.0;
                        break;
                    }
            }
        }


        // Calculate third eigenvector according to
        //   v[2] = v[0] x v[1]
        Q[0][2] = Q[1][0] * Q[2][1] - Q[2][0] * Q[1][1];
        Q[1][2] = Q[2][0] * Q[0][1] - Q[0][0] * Q[2][1];
        Q[2][2] = Q[0][0] * Q[1][1] - Q[1][0] * Q[0][1];
#endif

        return 0;
    }



    // ----------------------------------------------------------------------------
    __device__ void dsytrd3(float A[3][3], float Q[3][3], float d[3], float e[2])
        // ----------------------------------------------------------------------------
        // Reduces a symmetric 3x3 matrix to tridiagonal form by applying
        // (unitary) Householder transformations:
        //            [ d[0]  e[0]       ]
        //    A = Q . [ e[0]  d[1]  e[1] ] . Q^T
        //            [       e[1]  d[2] ]
        // The function accesses only the diagonal and upper triangular parts of
        // A. The access is read-only.
        // ---------------------------------------------------------------------------
    {
        const int n = 3;
        float u[n], q[n];
        float omega, f;
        float K, h, g;

        // Initialize Q to the identitity matrix
#ifndef EVALS_ONLY
        for (int i = 0; i < n; i++)
        {
            Q[i][i] = 1.0;
            for (int j = 0; j < i; j++)
                Q[i][j] = Q[j][i] = 0.0;
        }
#endif

        // Bring first row and column to the desired form 
        h = SQR(A[0][1]) + SQR(A[0][2]);
        if (A[0][1] > 0)
            g = -sqrt(h);
        else
            g = sqrt(h);
        e[0] = g;
        f = g * A[0][1];
        u[1] = A[0][1] - g;
        u[2] = A[0][2];

        omega = h - f;
        if (omega > 0.0)
        {
            omega = 1.0 / omega;
            K = 0.0;
            for (int i = 1; i < n; i++)
            {
                f = A[1][i] * u[1] + A[i][2] * u[2];
                q[i] = omega * f;                  // p
                K += u[i] * f;                   // u* A u
            }
            K *= 0.5 * SQR(omega);

            for (int i = 1; i < n; i++)
                q[i] = q[i] - K * u[i];

            d[0] = A[0][0];
            d[1] = A[1][1] - 2.0 * q[1] * u[1];
            d[2] = A[2][2] - 2.0 * q[2] * u[2];

            // Store inverse Householder transformation in Q
#ifndef EVALS_ONLY
            for (int j = 1; j < n; j++)
            {
                f = omega * u[j];
                for (int i = 1; i < n; i++)
                    Q[i][j] = Q[i][j] - f * u[i];
            }
#endif

            // Calculate updated A[1][2] and store it in e[1]
            e[1] = A[1][2] - q[1] * u[2] - u[1] * q[2];
        }
        else
        {
            for (int i = 0; i < n; i++)
                d[i] = A[i][i];
            e[1] = A[1][2];
        }
    }

    // ----------------------------------------------------------------------------
    __device__ int dsyevq3(float A[3][3], float Q[3][3], float w[3])
        // ----------------------------------------------------------------------------
        // Calculates the eigenvalues and normalized eigenvectors of a symmetric 3x3
        // matrix A using the QL algorithm with implicit shifts, preceded by a
        // Householder reduction to tridiagonal form.
        // The function accesses only the diagonal and upper triangular parts of A.
        // The access is read-only.
        // ----------------------------------------------------------------------------
        // Parameters:
        //   A: The symmetric input matrix
        //   Q: Storage buffer for eigenvectors
        //   w: Storage buffer for eigenvalues
        // ----------------------------------------------------------------------------
        // Return value:
        //   0: Success
        //  -1: Error (no convergence)
        // ----------------------------------------------------------------------------
        // Dependencies:
        //   dsytrd3()
        // ----------------------------------------------------------------------------
    {
        const int n = 3;
        float e[3];                   // The third element is used only as temporary workspace
        float g, r, p, f, b, s, c, t; // Intermediate storage
        int nIter;
        int m;

        // Transform A to float tridiagonal form by the Householder method
        dsytrd3(A, Q, w, e);

        // Calculate eigensystem of the remaining float symmetric tridiagonal matrix
        // with the QL method
        //
        // Loop over all off-diagonal elements
        for (int l = 0; l < n - 1; l++)
        {
            nIter = 0;
            while (1)
            {
                // Check for convergence and exit iteration loop if off-diagonal
                // element e(l) is zero
                for (m = l; m <= n - 2; m++)
                {
                    g = fabs(w[m]) + fabs(w[m + 1]);
                    if (fabs(e[m]) + g == g)
                        break;
                }
                if (m == l)
                    break;

                if (nIter++ >= 30)
                    return -1;

                // Calculate g = d_m - k
                g = (w[l + 1] - w[l]) / (e[l] + e[l]);
                r = sqrt(SQR(g) + 1.0);
                if (g > 0)
                    g = w[m] - w[l] + e[l] / (g + r);
                else
                    g = w[m] - w[l] + e[l] / (g - r);

                s = c = 1.0;
                p = 0.0;
                for (int i = m - 1; i >= l; i--)
                {
                    f = s * e[i];
                    b = c * e[i];
                    if (fabs(f) > fabs(g))
                    {
                        c = g / f;
                        r = sqrt(SQR(c) + 1.0);
                        e[i + 1] = f * r;
                        c *= (s = 1.0 / r);
                    }
                    else
                    {
                        s = f / g;
                        r = sqrt(SQR(s) + 1.0);
                        e[i + 1] = g * r;
                        s *= (c = 1.0 / r);
                    }

                    g = w[i + 1] - p;
                    r = (w[i] - g) * s + 2.0 * c * b;
                    p = s * r;
                    w[i + 1] = g + p;
                    g = c * r - b;

                    // Form eigenvectors
#ifndef EVALS_ONLY
                    for (int k = 0; k < n; k++)
                    {
                        t = Q[k][i + 1];
                        Q[k][i + 1] = s * Q[k][i] + c * t;
                        Q[k][i] = c * Q[k][i] - s * t;
                    }
#endif 
                }
                w[l] -= p;
                e[l] = g;
                e[m] = 0.0;
            }
        }

        return 0;
    }

    /////// __global__ interface,column major for matlab
    __device__ void eig2(const float* M, float* V, float* L) {
        dsyev2(M[0], M[1], M[3], &L[3], &L[0], &V[1], &V[3]);
        V[2] = V[1];
        V[0] = -V[3];
        L[1] = L[2] = 0;
    }

    __device__ void eig3(const float* M, float* V, float* L, bool useIterative = false) {

        float A[3][3] = { {M[0],M[1],M[2]},
                        {M[3],M[4],M[5]},
                        {M[6],M[7],M[8]} };

        float Q[3][3] = { {0,0,0},
                        {0,0,0},
                        {0,0,0} };

        float LL[3] = { 0,0,0 };

        int conv = 0;
        if (useIterative) {
            conv = dsyevq3(A, Q, LL);
        }
        else {
            conv = dsyevv3(A, Q, LL);
        }

        if (conv < 0) {
            L[0] = -1;
            return;
        }

        L[0] = LL[0];
        L[4] = LL[1];
        L[8] = LL[2];

        V[0] = Q[0][0]; V[1] = Q[1][0]; V[2] = Q[2][0];
        V[3] = Q[0][1]; V[4] = Q[1][1]; V[5] = Q[2][1];
        V[6] = Q[0][2]; V[7] = Q[1][2]; V[8] = Q[2][2];
    }

    __global__ void eig(const float* M, float* V, float* L, const int n, bool useIterative = false) {
        if (n == 2) {
            eig2(M, V, L);
        }
        else if (n == 3) {
            eig3(M, V, L, useIterative);
        }
    }

    __global__ void eigVal(const float* M, float* L, const int n) {
        if (n == 2) {
            dsyevc2(M[0], M[1], M[3], &L[1], &L[0]);
        }
        else if (n == 3) {
            float A[3][3] = { {M[0],M[1],M[2]},
                            {M[3],M[4],M[5]},
                            {M[6],M[7],M[8]} };
            dsyevc3(A, L);
        }
    }

}

namespace cuda
{
    struct ExtractPointCloud
    {
        Parameters parameters;
        pcl::gpu::PtrSz<float3> pointCloud;
        pcl::gpu::PtrSz<float3> pointCloudCache;
        pcl::gpu::PtrSz<float3> pointCloudNormals;
        pcl::gpu::PtrStepSz<int> indicesImage;
        pcl::gpu::PtrStepSz<uchar3> colorImage;
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
                printf("cx %f, cy %f, fx %f, fy %f, shift %f, width %d, height %d, boundary radius: %f, neighbourRadius: %d\n", 
                    parameters.cx, parameters.cy, parameters.fx, parameters.fy, parameters.depthShift, parameters.depthWidth, parameters.depthHeight, 
                    parameters.boundaryEstimationDistance, parameters.neighbourRadius);
            }

            int ix = index % parameters.depthWidth;
            int iy = index / parameters.depthWidth;

            float x = 0, y = 0, z = 0;
            float zValue = depthImage[index] * 1.f;
            z = zValue / parameters.depthShift;
            x = (ix - parameters.cx) * z / parameters.fx;
            y = (iy - parameters.cy) * z / parameters.fy;

            //const float qnan = std::numeric_limits<float>::quiet_NaN();

            //if (index % 1024 == 0)
            //{
                //printf("index: %d, ix: %d, iy: %d, x: %f, y: %f, z: %f, depth: %d\n", index, ix, iy, x, y, z, depthImage[index]);
            //}
            pointCloud[index].x = x;
            pointCloud[index].y = y;
            pointCloud[index].z = z;

            if (isValid(pointCloud[index]))
            {
                indicesImage[index] = index;
            }
        }

        __device__ void holeFilling()
        {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            int ix = index % parameters.depthWidth;
            int iy = index / parameters.depthWidth;

            float x = 0, y = 0, z = 0;
            int count = 0;

            float3 point = pointCloud[index];
            if (isValid(point))
                return;

            uint* indices = (uint*)(neighbours + index * (parameters.neighbourRadius * parameters.neighbourRadius + 1));
            if (indices[0] * 1.0f / (parameters.neighbourRadius * parameters.neighbourRadius) < 0.3f)
            {
                return;
            }

            for (int i = 1; i <= indices[0]; i++)
            {
                size_t neighbourIndex = indices[i];
                float3 neighbourPoint = pointCloud[neighbourIndex];
                if (!isValid(neighbourPoint))
                    continue;

                z += neighbourPoint.z;
                count++;
            }
            z /= count;

            x = (ix - parameters.cx) * z / parameters.fx;
            y = (iy - parameters.cy) * z / parameters.fy;

            //const float qnan = std::numeric_limits<float>::quiet_NaN();

            //if (index % 1024 == 0)
            //{
                //printf("index: %d, ix: %d, iy: %d, x: %f, y: %f, z: %f, depth: %d\n", index, ix, iy, x, y, z, depthImage[index]);
            //}
            pointCloud[index].x = x;
            pointCloud[index].y = y;
            pointCloud[index].z = z;

            if (isValid(pointCloud[index]))
            {
                indicesImage[index] = index;
            }
        }

        __device__ void estimateNormals()
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

            //uint* indices = (uint*)(neighbours + index * (parameters.neighbourRadius * parameters.neighbourRadius + 1));
            //indices[0] = 0;

            float maxDistance = parameters.normalKernelMaxDistance * parameters.normalKernelMaxDistance;

            for (int j = max(0, iy - parameters.neighbourRadius / 2); j < min(iy + parameters.neighbourRadius / 2 + 1, parameters.depthHeight); j++) {
                for (int i = max(0, ix - parameters.neighbourRadius / 2); i < min(ix + parameters.neighbourRadius / 2 + 1, parameters.depthWidth); i++) {
                    size_t neighbourIndex = j * parameters.depthWidth + i;
                    float3 diff3 = pointCloud[neighbourIndex] - pointCloud[index];
                    float norm = dot(diff3, diff3);
                    if (norm < maxDistance) {
                        center += pointCloud[neighbourIndex];
                        outerAdd(pointCloud[neighbourIndex], C); //note: instead of pts[ind]-center, we demean afterwards
                        count++;
                        //indices[count] = static_cast<uint>(neighbourIndex);
                    }
                }
            }
            //indices[0] = count;
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

            pointCloudNormals[index].x = normal.x;
            pointCloudNormals[index].y = normal.y;
            pointCloudNormals[index].z = normal.z;
        }

        __device__ void extractBoundaries()
        {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            int ix = index % parameters.depthWidth;
            int iy = index / parameters.depthWidth;

            /*uint* indices = (uint*)(neighbours + index * (parameters.neighbourRadius * parameters.neighbourRadius + 1));
            if (indices[0] == 0)
                return;*/

            int beAngles[360] = { 0 };
            float cornerAngles[360] = { 0 };
            float cornerValues[360] = { 0 };
            int count = 0;

            float3 point = pointCloud[index];
            if (!isValid(point))
                return;

            float maxDistance = parameters.normalKernelMaxDistance * parameters.normalKernelMaxDistance;

            float3 normal = pointCloudNormals[index];
            float3 beU, beV;
            beV = pcl::cuda::unitOrthogonal(normal);
            beU = cross(normal, beV);

            float3 cornerU, cornerV;
            cornerU.x = cornerU.y = cornerU.z = 0;
            float maxCornerAngle = 0;

            for (int j = max(0, iy - parameters.neighbourRadius / 2); j < min(iy + parameters.neighbourRadius / 2 + 1, parameters.depthHeight); j++) {
                for (int i = max(0, ix - parameters.neighbourRadius / 2); i < min(ix + parameters.neighbourRadius / 2 + 1, parameters.depthWidth); i++) {
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

            for (int j = max(0, iy - parameters.neighbourRadius / 2); j < min(iy + parameters.neighbourRadius / 2 + 1, parameters.depthHeight); j++) {
                for (int i = max(0, ix - parameters.neighbourRadius / 2); i < min(ix + parameters.neighbourRadius / 2 + 1, parameters.depthWidth); i++) {
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
                        if (diff > sigma * 1.f && cornerValues[i] > avgValue)
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
                        if (nextIndex - index > 5)
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

                if (ix == parameters.debugX && iy == parameters.debugY)
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
                }

                if (count > 1 && clusterPeaks >= 2 && clusterPeaks <= 3)
                {
                    if (!(ix <= parameters.borderLeft || iy <= parameters.borderTop || ix >= (parameters.depthWidth - parameters.borderRight) || iy >= (parameters.depthHeight - parameters.borderBottom)))
                    {
                        boundaries[index] = 4;
                        boundaryImage[index] = 4;
                    }
                }
            }
        }

        __device__ void classifyBoundaries()
        {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            int ix = index % parameters.depthWidth;
            int iy = index / parameters.depthWidth;

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
                            //if (index % 500 == 0)
                                //printf("index: %d, ix: %d, iy: %d, dist: %f\n", index, ix, iy, dist);
                            //if (dist >= 0.8f && point.z > nPt.z)
                            if ((point.z - nPt.z) >= 0.1f)
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

        __device__ void extractCornerPoints()
        {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            int ix = index % parameters.depthWidth;
            int iy = index / parameters.depthWidth;

            float cornerAngles[360] = { 0 };
            float cornerValues[360] = { 0 };

            uint* indices = (uint*)(neighbours + index * (parameters.neighbourRadius * parameters.neighbourRadius + 1));
            if (indices[0] == 0)
                return;

            float3 point = pointCloud[index];
            if (!isValid(point))
                return;

            float3 normal = pointCloudNormals[index];
            float3 cornerU, cornerV;
            cornerU.x = cornerU.y = cornerU.z = 0;

            int count = 0;
            
            for (int i = 1; i <= indices[0]; i++)
            {
                size_t neighbourIndex = indices[i];
                float3 neighbourPoint = pointCloud[neighbourIndex];
                float3 neighbourNormal = pointCloudNormals[neighbourIndex];
                if (!isValid(neighbourPoint))
                    continue;

                cornerU += neighbourNormal;
                count++;
            }
            cornerU = normalize(cornerU / count);
            cornerV = cross(cross(cornerU, normal), normal);

            for (int i = 1; i <= indices[0]; i++)
            {
                size_t neighbourIndex = indices[i];
                float3 neighbourPoint = pointCloud[neighbourIndex];
                float3 neighbourNormal = pointCloudNormals[neighbourIndex];
                if (!isValid(neighbourPoint))
                    continue;

                float angle = atan2(dot(cornerV, neighbourNormal), dot(cornerU, neighbourNormal));
                int angleIndex = (int)(angle * 180 / M_PI) + 180;
                cornerAngles[angleIndex] += 1;
            }

            float maxValue;
            float avgValue = 0;
            count = 0;
            // 1-dimension filter
            float kernel[5] = { -1, -1, 5, -1, -1 };
            for (int i = 0; i < 360; i++)
            {
                if (cornerAngles[i] <= 0)
                    continue;

                float value = cornerAngles[i] * cornerAngles[i];
                
                if (value < 0)
                    value = 0;
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
                for (int i = 0; i < 360; i++)
                {
                    if (cornerValues[i] > 0)
                    {
                        float value = cornerValues[i];
                        if (value < 0)
                        {
                            value = 0;
                        }
                        else
                        {
                            count++;
                        }
                        if (maxValue < value)
                        {
                            maxValue = value;
                        }
                        cornerValues[i] = value;
                    }
                }

                avgValue = 0;
                for (int i = 0; i < 360; i++)
                {
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
                    if (diff > sigma * 0.5f && cornerValues[i] > avgValue)
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
                for (int i = 0; i < peaks - 1; i++)
                {
                    int index = static_cast<int>(cornerAngles[i]);
                    int nextIndex = static_cast<int>(cornerAngles[i + 1]);
                    if (ix == parameters.debugX && iy == parameters.debugY)
                        printf("  %d %d %d %d\n", i, index, nextIndex, clusterPeaks);
                    if (nextIndex - index > 5)
                    {
                        clusterPeaks++;
                    }
                }
            }

            if (ix == parameters.debugX && iy == parameters.debugY)
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
            }

            if (count > 1 && clusterPeaks >= 2 && clusterPeaks <=2)
            {
                
                if (!(ix <= parameters.borderLeft || iy <= parameters.borderTop || ix >= (parameters.depthWidth - parameters.borderRight) || iy >= (parameters.depthHeight - parameters.borderBottom)))
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
                }
            }
        }
    };

    __global__ void extractPointCloud(ExtractPointCloud epc)
    {
        epc.extracPointCloud();
    }

    __global__ void holeFilling(ExtractPointCloud epc)
    {
        epc.holeFilling();
    }

    __global__ void estimateNormals(ExtractPointCloud epc)
    {
        epc.estimateNormals();
    }

    __global__ void extractBoundaries(ExtractPointCloud epc)
    {
        epc.extractBoundaries();
    }

    __global__ void smoothBoudaries(ExtractPointCloud epc)
    {
        //epc.smoothBoundaries();
    }

    __global__ void classifyBoundaries(ExtractPointCloud epc)
    {
        epc.classifyBoundaries();
    }

    __global__ void extractCornerPoints(ExtractPointCloud epc)
    {
        epc.extractCornerPoints();
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
        safeCall(cudaDeviceSynchronize());

        ExtractPointCloud epc;
        epc.parameters = frame.parameters;
        epc.pointCloud = frame.pointCloud;
        epc.pointCloudCache = frame.pointCloudCache;
        epc.pointCloudNormals = frame.pointCloudNormals;
        epc.indicesImage = frame.indicesImage;
        epc.colorImage = frame.colorImage;
        epc.depthImage = frame.depthImage;
        epc.boundaries = frame.boundaries;
        epc.boundaryImage = frame.boundaryImage;
        epc.neighbours = frame.neighbours;

        TICK("cuda_extractPointCloud");
        extractPointCloud<<<grid, block>>>(epc);
        safeCall(cudaDeviceSynchronize());
        TOCK("cuda_extractPointCloud");

        //holeFilling<<<grid, block>>>(epc);
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

        //TICK("cuda_extractCornerPoints");
        //extractCornerPoints<<<grid, block>>>(epc);
        //safeCall(cudaDeviceSynchronize());
        //TOCK("cuda_extractCornerPoints");
    }
}