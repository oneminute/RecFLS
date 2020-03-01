#include "CudaInternal.h"
//#include "cutil_math.h"
#include "cutil_math2.h"

#include <QtMath>
#include <opencv2/opencv.hpp>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/cuda/common/eigen.h>
#include <math.h>
#include <float.h>

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

    struct ExtractPointCloud
    {
        Parameters parameters;
        pcl::gpu::PtrSz<float3> points;
        pcl::gpu::PtrSz<float3> normals;
        pcl::gpu::PtrStepSz<int> indicesImage;
        pcl::gpu::PtrStepSz<uchar3> colorImage;
        pcl::gpu::PtrStepSz<ushort> depthImage;
        pcl::gpu::PtrSz<uchar> boundaryIndices;
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
            points[index].x = x;
            points[index].y = y;
            points[index].z = z;

            //boundaries[index].x = 0;
            //boundaries[index].y = 0;
            //boundaries[index].z = 0;

            if (isValid(points[index]))
            {
                indicesImage[index] = index;
            }
        }

        __device__ void fillZeros(float** m, int rows, int cols)
        {
            for (int i = 0; i < rows; i++)
            {
                for (int j = 0; j < cols; j++)
                {
                    m[i][j] = 0;
                }
            }
        }

        __device__ void estimateNormals()
        {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            int ix = index % parameters.depthWidth;
            int iy = index / parameters.depthWidth;

            float3 m = make_float3(0, 0, 0);
            float C_2d[3][3] = { {0,0,0},{0,0,0},{0,0,0} };
            float* C = (float*)C_2d; //rowwise
            float3 normal;

            uint* indices = (uint*)(neighbours + index * (parameters.neighbourRadius * parameters.neighbourRadius + 1));
            indices[0] = 0;

            // first iteration
            int count = 0;
            for (int j = max(0, iy - parameters.neighbourRadius / 2); j < min(iy + parameters.neighbourRadius / 2 + 1, parameters.depthHeight); j++) {
                for (int i = max(0, ix - parameters.neighbourRadius / 2); i < min(ix + parameters.neighbourRadius / 2 + 1, parameters.depthWidth); i++) {
                    size_t neighbourIndex = j * parameters.depthWidth + i;
                    float3 diff3 = points[neighbourIndex] - points[index];
                    float norm = sqrt(dot(diff3, diff3));
                    if (norm < parameters.normalKernelMaxDistance) {
                        m += points[neighbourIndex];
                        outerAdd(points[neighbourIndex], C); //note: instead of pts[ind]-m, we demean afterwards
                        count++;
                        indices[count] = static_cast<uint>(neighbourIndex);
                    }
                }
            }
            indices[0] = count;
            if (count < 3)
                return;

            m /= (count + 0.0f);
            outerAdd(m, C, -count);
            float fac = 1.0f / (count - 1.0f);
            mul(C, fac);
            float Q[3][3] = { {0,0,0},{0,0,0},{0,0,0} };
            float w[3] = { 0, 0, 0 };
            int result = dsyevv3(C_2d, Q, w);

            //the largest eigenvector is the rightmost column
            normal.x = Q[0][0];
            normal.y = Q[1][0];
            normal.z = Q[2][0];
            // end first iteration

            normals[index].x = normal.x;
            normals[index].y = normal.y;
            normals[index].z = normal.z;

            float stdDiv = 0;

            //if (index % 1024 == 0)
                //printf("ix: %3d, iy: %3d, count: %4d, result: %d, stdDiv: %f, %f, %f, %f\n", ix, iy, count, result, stdDiv, w[0], w[1], w[2]);

            //int iteration = 1;
            //bool candidate = false;
            //while (count > 20 && iteration < 8)
            ////while (false)
            //{
            //    for (int i = 0; i < 3; i++)
            //    {
            //        for (int j = 0; j < 3; j++)
            //        {
            //            C_2d[i][j] = 0;
            //            Q[i][j] = 0;
            //        }
            //    }
            //    float avgD = 0;
            //    float3 center = m;
            //    for (int i = 1; i <= count; i++)
            //    {
            //        float3 neighbourPoint = points[indices[i]];
            //        float3 diff = neighbourPoint - center;
            //        float dist = abs(dot(diff, normal));
            //        avgD += dist;
            //    }
            //    avgD /= (count + 0.0f);
            //    float sigma = 0;
            //    for (int i = 1; i <= count; i++)
            //    {
            //        float3 neighbourPoint = points[indices[i]];
            //        float3 diff = neighbourPoint - center;
            //        float dist = abs(dot(diff, normal));
            //        float distDiff = dist - avgD;
            //        sigma += distDiff * distDiff;
            //    }
            //    sigma = sqrt(sigma / (count - 1.0f));

            //    //m = make_float3(0, 0, 0);
            //    m.x = m.y = m.z = 0.0f;
            //    int tmpCount = 0;
            //   //if (ix == parameters.debugX && iy == parameters.debugY)
            //        //printf("%3d %3d, iteration: %2d, count: %4d, sigma: %f\n", ix, iy, iteration, count, sigma);
            //    for (int i = 1; i <= count; i++)
            //    {
            //        float3 diff = points[indices[i]] - center;
            //        float dist = abs(dot(diff, normal));
            //        float distDiff = dist - avgD;
            //        if (distDiff < sigma * 2)
            //        {
            //        //    //printf("count: %d, tmpCount: %d, sigma: %f, %f %f %f\n", count, tmpCount, sigma, neighbourPoint.x, neighbourPoint.y, neighbourPoint.z);
            //            outerAdd(points[indices[i]], C);
            //            m += points[indices[i]];
            //            tmpCount++;
            //            indices[tmpCount] = indices[i];
            //        }
            //    }
            //    ////count = tmpCount;

            //    //if (tmpCount < 20)
            //    //{
            //    //    candidate = false;
            //    //    count = tmpCount;
            //    //    break;
            //    //}

            //    //m /= (tmpCount + 0.0f);
            //    //outerAdd(m, C, -tmpCount);
            //    //m = center;
            //    //float fac = 1.0f / (tmpCount - 1.0f);
            //    //mul(C, fac);
            //    //result = dsyevv3(C_2d, Q, w);
            //    //stdDiv = sigma;
            //    //normal.x = Q[0][0];
            //    //normal.y = Q[1][0];
            //    //normal.z = Q[2][0];

            //    //if (result == -1)
            //    //{
            //    //    candidate = false;
            //    //    break;
            //    //}

            //    //if (tmpCount == count)
            //    //{
            //    //    count = tmpCount;
            //    //    break;
            //    //}
            //    //count = tmpCount;
            //    iteration++;
            //}

            //if (candidate)
            //{
            //    if (abs(dot((points[index] - m), normal)) < 2 * stdDiv)
            //    {
            //        for (int i = 0; i < 3; i++)
            //        {
            //            for (int j = 0; j < 3; j++)
            //            {
            //                C_2d[i][j] = 0;
            //                Q[i][j] = 0;
            //            }
            //        }
            //        float avgD = 0;
            //        float3 center = m;
            //        for (int i = 1; i < count + 1; i++)
            //        {
            //            float3 neighbourPoint = points[indices[i]];
            //            float3 diff = neighbourPoint - center;
            //            float dist = abs(dot(diff, normal));
            //            avgD += dist;
            //        }
            //        avgD /= count;
            //        m = make_float3(0, 0, 0);
            //        int tmpCount = count;
            //        count = 0;
            //        for (int j = max(0, iy - parameters.neighbourRadius / 2); j < min(iy + parameters.neighbourRadius / 2 + 1, parameters.depthHeight); j++) {
            //            for (int i = max(0, ix - parameters.neighbourRadius / 2); i < min(ix + parameters.neighbourRadius / 2 + 1, parameters.depthWidth); i++) {
            //                size_t neighbourIndex = j * parameters.depthWidth + i;
            //                float3 neighbourPoint = points[neighbourIndex];
            //                float3 diff3 = points[neighbourIndex] - points[index];
            //                float norm = sqrt(dot(diff3, diff3));
            //                if (norm < parameters.normalKernelMaxDistance) {
            //                    float3 diff = neighbourPoint - center;
            //                    float dist = abs(dot(diff, normal));
            //                    float distDiff = dist - avgD;
            //                    if (distDiff < stdDiv * 2)
            //                    {
            //                        m += neighbourPoint;
            //                        count++;
            //                        indices[count] = static_cast<uint>(neighbourIndex);
            //                        outerAdd(neighbourPoint, C);
            //                    }
            //                }
            //            }
            //        }

            //        if (count >= 20)
            //        {
            //            m /= (count + 0.0f);
            //            outerAdd(m, C, -count);

            //            float fac = 1.0f / (count - 1.0f);
            //            mul(C, fac);

            //            result = dsyevv3(C_2d, Q, w);
            //            if (result == -1)
            //            {
            //                candidate = false;
            //            }
            //            else
            //            {
            //                normal.x = Q[0][0];
            //                normal.y = Q[1][0];
            //                normal.z = Q[2][0];
            //            }

            //            if (index % 1024 == 0)
            //                printf("ix: %3d, iy: %3d, iteration: %2d, candidate: %d, count: %4d, stdDiv: %f, %f, %f, %f\n", ix, iy, iteration, candidate, count, stdDiv, w[0], w[1], w[2]);
            //        }
            //        else
            //        {
            //            candidate = false;
            //        }

            //        if (candidate)
            //        {
            //            indices[0] = count;
            //            normals[index].x = normal.x;
            //            normals[index].y = normal.y;
            //            normals[index].z = normal.z;
            //        }
            //    }
            //    else
            //    {
            //        candidate = false;
            //    }
            //}
            //else
            //{
            //    indices[0] = 0;
            //}

            //if (index % 1024 == 0)
                //printf("ix: %3d, iy: %3d, iteration: %2d, candidate: %d, count: %4d, stdDiv: %f, %f, %f, %f\n", ix, iy, iteration, candidate, count, stdDiv, w[0], w[1], w[2]);

        }

        __device__ void extractBoundaries()
        {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            int ix = index % parameters.depthWidth;
            int iy = index / parameters.depthWidth;

            uint* indices = (uint*)(neighbours + index * (parameters.neighbourRadius * parameters.neighbourRadius + 1));
            if (indices[0] == 0)
                return;

            int angles[360] = { 0 };
            int count = 0;

            float3 point = points[index];
            if (!isValid(point))
                return;

            float3 normal = normals[index];
            float3 u, v;
            v = pcl::cuda::unitOrthogonal(normal);
            u = cross(normal, v);

            int maxAngleIndex = -1000;

            ////for (int j = max(0, iy - parameters.boundaryEstimationRadius); j < min(iy + parameters.boundaryEstimationRadius + 1, parameters.depthHeight); j++) 
            //{
            //    for (int i = max(0, ix - parameters.boundaryEstimationRadius); i < min(ix + parameters.boundaryEstimationRadius + 1, parameters.depthWidth); i++) 
            //    {
            //        size_t neighbourIndex = j * parameters.depthWidth + i;
            //        float3 neighbourPoint = points[neighbourIndex];
            //        if (!isValid(neighbourPoint))
            //            continue;

            //        float3 delta = neighbourPoint - point;
            //        float distance = sqrt(dot(delta, delta));

            //        float thresh = parameters.boundaryEstimationDistance;
            //        if (neighbourPoint.z >= 1)
            //        {
            //            thresh *= neighbourPoint.z;
            //        }
            //        if (distance < thresh) 
            //        {
            //            float angle = atan2(dot(v, delta), dot(u, delta));
            //            int angleIndex = (int)(angle * 180 / M_PI) + 180;
            //            if (maxAngleIndex < angleIndex)
            //                maxAngleIndex = angleIndex;
            //            angles[angleIndex] += 1;
            //            count++;
            //        }
            //    }
            //}
            for (int i = 1; i <= indices[0]; i++)
            {
                size_t neighbourIndex = indices[i];
                float3 neighbourPoint = points[neighbourIndex];
                if (!isValid(neighbourPoint))
                    continue;

                float3 delta = neighbourPoint - point;
                float angle = atan2(dot(v, delta), dot(u, delta));
                int angleIndex = (int)(angle * 180 / M_PI) + 180;
                if (maxAngleIndex < angleIndex)
                    maxAngleIndex = angleIndex;
                angles[angleIndex] += 1;
                count++;
            }

            int angleDiff = 0;
            int maxDiff = 0;
            int lastAngle = 0;
            int firstAngle = lastAngle;
            for (int i = 1; i < 360; i++)
            {
                if (angles[i] == 0)
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

            //if (index % 1024 == 0)
            //{
                //printf("index: %d, ix: %d, iy: %d, count: %d, maxAngleIndex: %d, maxDiff: %d, pxX: %d, pxY: %d\n", index, ix, iy, count, maxAngleIndex, maxDiff, pxX, pxY);
            //}

            if (maxDiff > parameters.boundaryAngleThreshold)
            {
                if (pxX <= parameters.borderLeft || pxY <= parameters.borderTop || pxX >= (parameters.depthWidth - parameters.borderRight) || pxY >= (parameters.depthHeight - parameters.borderBottom))
                {
                    boundaryIndices[index] = 1;
                    boundaryImage[pxY * parameters.depthWidth + pxX] = 1;
                }
                else
                {
                    boundaryIndices[index] = 3;
                    boundaryImage[pxY * parameters.depthWidth + pxX] = 3;
                    //boundaryIndicesMat[pxY * parameters.depthWidth + pxX] = index;
                }
            }
        }

        __device__ void extractBoundaries2()
        {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            int ix = index % parameters.depthWidth;
            int iy = index / parameters.depthWidth;

            int angles[360] = { 0 };
            int count = 0;

            uint* indices = (uint*)(neighbours + index * (parameters.neighbourRadius * parameters.neighbourRadius + 1));
            if (indices[0] == 0)
                return;

            float3 point = points[index];
            if (!isValid(point))
                return;

            float3 normal = normals[index];
            float3 u, v;
            v = pcl::cuda::unitOrthogonal(normal);
            u = cross(normal, v);

            int maxAngleIndex = -1000;

            for (int i = 1; i <= indices[0]; i++)
            {
                size_t neighbourIndex = indices[i];
                float3 neighbourPoint = points[neighbourIndex];
                if (!isValid(neighbourPoint))
                    continue;

                float3 delta = neighbourPoint - point;
                float angle = atan2(dot(v, delta), dot(u, delta));
                int angleIndex = (int)(angle * 180 / M_PI) + 180;
                if (maxAngleIndex < angleIndex)
                    maxAngleIndex = angleIndex;
                angles[angleIndex] += 1;
                count++;
            }

            int angleDiff = 0;
            int maxDiff = 0;
            int lastAngle = 0;
            int firstAngle = lastAngle;
            for (int i = 1; i < 360; i++)
            {
                if (angles[i] == 0)
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

            //if (index % 1024 == 0)
            //{
                //printf("index: %ld, ix: %d, iy: %d, count: %d, maxAngleIndex: %d, maxDiff: %d, pxX: %d, pxY: %d\n", index, ix, iy, count, maxAngleIndex, maxDiff, pxX, pxY);
            //}

            if (maxDiff > parameters.boundaryAngleThreshold)
            {
                if (pxX <= parameters.borderLeft || pxY <= parameters.borderTop || pxX >= (parameters.depthWidth - parameters.borderRight) || pxY >= (parameters.depthHeight - parameters.borderBottom))
                {
                    boundaryIndices[index] = 1;
                    boundaryImage[pxY * parameters.depthWidth + pxX] = 1;
                }
                else
                {
                    boundaryIndices[index] = 3;
                    boundaryImage[pxY * parameters.depthWidth + pxX] = 3;
                    //boundaryIndicesMat[pxY * parameters.depthWidth + pxX] = index;
                }
            }
        }

        __device__ void smoothBoundaries()
        {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            int ix = index % parameters.depthWidth;
            int iy = index / parameters.depthWidth;

            if (boundaryImage[index] < 1)
                return;

            int pointIndex = indicesImage[index];
            //if (ix == iy) printf("ix: %3d, iy: %3d, index: %6d, pointType: %d %d, pointIndex: %d\n", ix, iy, index, boundaryImage[index], pointIndex);

            if (pointIndex <= 0)
                return;

            float3 point = points[pointIndex];
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
                    float3 neighbourPoint = points[neighbourIndex];

                    if (boundaryImage[neighbourIndex] <= 1)
                        continue;

                    if (!isValid(neighbourPoint))
                        continue;

                    float3 diff = point - neighbourPoint;
                    float dist = sqrt(dot(diff, diff));
                    if (dist < parameters.classifyDistance)
                    {
                        float g = 1 / (2 * M_PI * sigma * sigma) * powf(M_E, -((i - ix) * (i - ix) + (j - iy) * (j - iy)) / (2 * sigma * sigma));
                        value += g * neighbourPoint.z;
                        sumG += g;
                    }
                }
            }
            value /= sumG;
            if (isnan(value))
                return;

            //if (pointIndex % 20 == 0) printf("ix: %3d, iy: %3d, index: %6d, pointType: %d %d, pointIndex: %6d, value: %f: z: %f\n", ix, iy, index, boundaryImage[index], pointIndex, value, point.z);

            points[pointIndex].z = value;
        }

        __device__ void classifyBoundaries()
        {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            int ix = index % parameters.depthWidth;
            int iy = index / parameters.depthWidth;

            /*if (index % 1000 == 0)
            {
                printf("index: %d, ix: %d, iy: %d, boundary type: %d, point index: %d, point z: %f\n", index, ix, iy, boundaryImage[index], boundaryIndices[index], points[boundaryIndices[index]].z);
            }*/

            if (boundaryImage[index] <= 1)
                return;

            //int pointIndex = boundaryIndicesMat[index];
            //if (pointIndex <= 0)
                //return;

            float3 point = points[index];
            if (!isValid(point))
                return;

            float2 original = { parameters.cx, parameters.cy };
            float2 coord = { static_cast<float>(ix), static_cast<float>(iy) };

            float2 ray = normalize(original - coord);

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
                        if (boundaryIndices[nPxIndex] <= 0)
                            continue;

                        float3 nPt = points[nPxIndex];
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
                boundaryIndices[index] = 2;
                boundaryImage[index] = 2;
            }
            
            //if (index % 500 == 0)
                //printf("index: %d, ix: %d, iy: %d, ori.x: %f, ori.y: %f, ray.x: %f, ray.y: %f, veil: %d, radius: %d\n", index, ix, iy, original.x, original.y, ray.x, ray.y, veil, parameters.classifyRadius);
        }

        __device__ void extractCornerPoints()
        {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            int ix = index % parameters.depthWidth;
            int iy = index / parameters.depthWidth;

            float angles[360] = { 0 };
            float values[360] = { 0 };

            uint* indices = (uint*)(neighbours + index * (parameters.neighbourRadius * parameters.neighbourRadius + 1));
            if (indices[0] == 0)
                return;

            float3 point = points[index];
            if (!isValid(point))
                return;

            float3 normal = normals[index];
            float3 u, v;

            u.x = u.y = u.z = 0;

            int count = 0;
            /*for (int j = max(0, iy - parameters.boundaryEstimationRadius); j < min(iy + parameters.boundaryEstimationRadius + 1, parameters.depthHeight); j++) 
            {
                for (int i = max(0, ix - parameters.boundaryEstimationRadius); i < min(ix + parameters.boundaryEstimationRadius + 1, parameters.depthWidth); i++) 
                {
                    size_t neighbourIndex = j * parameters.depthWidth + i;
                    float3 neighbourPoint = points[neighbourIndex];
                    float3 neighbourNormal = normals[neighbourIndex];
                    if (!isValid(neighbourPoint))
                        continue;

                    float3 delta = neighbourPoint - point;
                    float distance = sqrt(dot(delta, delta));

                    float thresh = parameters.normalKernelMaxDistance;
                    if (neighbourPoint.z >= 1)
                    {
                        thresh *= neighbourPoint.z;
                    }
                    if (distance < thresh)
                    {
                        count++;
                        u += neighbourNormal;
                    }
                }
            }*/
            for (int i = 1; i <= indices[0]; i++)
            {
                size_t neighbourIndex = indices[i];
                float3 neighbourPoint = points[neighbourIndex];
                float3 neighbourNormal = normals[neighbourIndex];
                if (!isValid(neighbourPoint))
                    continue;

                u += neighbourNormal;
                count++;
            }
            u = normalize(u / count);
            v = cross(cross(u, normal), normal);

            float maxValue;
            for (int i = 1; i <= indices[0]; i++)
            {
                size_t neighbourIndex = indices[i];
                float3 neighbourPoint = points[neighbourIndex];
                float3 neighbourNormal = normals[neighbourIndex];
                if (!isValid(neighbourPoint))
                    continue;

                float angle = atan2(dot(v, neighbourNormal), dot(u, neighbourNormal));
                int angleIndex = (int)(angle * 180 / M_PI) + 180;
                angles[angleIndex] += 1;
            }

            float avgValue = 0;
            count = 0;
            // 1-dimension filter
            float kernel[5] = { -1, -1, 5, -1, -1 };
            for (int i = 0; i < 360; i++)
            {
                if (angles[i] <= 0)
                    continue;

                float value = angles[i] * angles[i];
                /*for (int j = -2; j <= 2; j++)
                {
                    float weight = kernel[j + 2];
                    int index = i + j;

                    if (index < 0)
                        index += 360;

                    if (index >= 360)
                        index -= 360;

                    value += angles[index] * weight;
                }*/
                if (value < 0)
                    value = 0;
                values[i] = value;
                avgValue += values[i];
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
                    if (values[i] > 0)
                    {
                        values[i] = values[i] / maxValue;
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
                    if (values[i] > 0)
                    {
                        float value = values[i];// -avgValue;
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
                        values[i] = value;
                    }
                }

                avgValue = 0;
                for (int i = 0; i < 360; i++)
                {
                    values[i] = values[i] / maxValue;
                    avgValue += values[i];

                    if (values[i] > 0)
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
                    if (values[i] <= 0)
                        continue;

                    float diff = abs(values[i] - avgValue);
                    sigma += diff * diff;
                    count += 1;
                }
                sigma = sqrt(sigma / count);

                float avgPeak = 0;
                for (int i = 0; i < 360; i++)
                {
                    float diff = abs(values[i] - avgValue);
                    angles[i] = -1;
                    if (diff > sigma * 0.5f && values[i] > avgValue)
                    //if (values[i] > avgValue)
                    {
                        angles[peaks] = i;
                        if (ix == parameters.debugX && iy == parameters.debugY)
                            printf("  %f %f %f %f\n", angles[i], values[i], diff, avgValue);
                        peaks++;
                    }
                }

                if (peaks > 0)
                    clusterPeaks = 1;
                for (int i = 0; i < peaks - 1; i++)
                {
                    int index = static_cast<int>(angles[i]);
                    int nextIndex = static_cast<int>(angles[i + 1]);
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
                    if (i == angles[peakCount])
                    {
                        isPeak = true;
                        peakCount++;
                    }
                    printf("    %4d: %2.6f %d ", i, values[i], isPeak);
                    int pluses = ceil(values[i] / 0.05f);
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
                
                //printf("ix: %4ld, iy: %4ld, count: %2d, sigma: %3.6f, peaks: %2d, avgValue: %f, maxValue: %f\n", 
                    //ix, iy, count, sigma, peaks, avgValue, maxValue);
                //printf("ix: %3d, iy: %3d, count: %4d, sigma: %f, peaks: %2d, avgValue: %f, maxValue: %f\n", ix, iy, count, sigma, peaks, avgValue, maxValue);
                //printf("ix: %3d, iy: %3d, count: %4d, maxDiff: %4d, first angle: %3d, last angle: %3d\n", ix, iy, count, maxDiff, firstAngle, lastAngle);
                if (!(ix <= parameters.borderLeft || iy <= parameters.borderTop || ix >= (parameters.depthWidth - parameters.borderRight) || iy >= (parameters.depthHeight - parameters.borderBottom)))
                {
                    boundaryIndices[index] = 4;
                    boundaryImage[index] = 3 + clusterPeaks * 20;
                }
            }
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

    __global__ void smoothBoudaries(ExtractPointCloud epc)
    {
        epc.smoothBoundaries();
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
        dim3 grid(frame.parameters.depthWidth * frame.parameters.depthHeight / 256);
        dim3 block(256);

        int size = frame.parameters.depthWidth * frame.parameters.depthHeight;
        cudaMemset(frame.pointCloud.ptr(), 0, size * sizeof(float3));
        cudaMemset(frame.pointCloudNormals.ptr(), 0, size * sizeof(float3));
        cudaMemset(frame.indicesImage.ptr(), 0, size * sizeof(int));
        cudaMemset(frame.boundaries.ptr(), 0, size * sizeof(uchar));
        cudaMemset(frame.boundaryImage.ptr(), 0, size * sizeof(uchar));
        safeCall(cudaDeviceSynchronize());

        ExtractPointCloud epc;
        epc.parameters = frame.parameters;
        epc.points = frame.pointCloud;
        epc.normals = frame.pointCloudNormals;
        epc.indicesImage = frame.indicesImage;
        epc.colorImage = frame.colorImage;
        epc.depthImage = frame.depthImage;
        epc.boundaryIndices = frame.boundaries;
        epc.boundaryImage = frame.boundaryImage;
        epc.neighbours = frame.neighbours;

        extractPointCloud<<<grid, block>>>(epc);
        safeCall(cudaDeviceSynchronize());

        estimateNormals<<<grid, block>>>(epc);
        safeCall(cudaDeviceSynchronize());

        extractBoundaries<<<grid, block>>>(epc);
        safeCall(cudaDeviceSynchronize());

        //smoothBoudaries<<<grid, block>>>(epc);
        //safeCall(cudaDeviceSynchronize());

        classifyBoundaries<<<grid, block>>>(epc);
        safeCall(cudaDeviceSynchronize());

        extractCornerPoints<<<grid, block>>>(epc);
        safeCall(cudaDeviceSynchronize());
    }
}