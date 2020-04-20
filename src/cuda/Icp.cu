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

#define SQR(x)      ((x)*(x))                        // x^2 
#define SQR_ABS(x)  (SQR(cfloat(x)) + SQR(cimag(x)))  // |x|^2
#define M_SQRT3    1.73205080756887729352744634151   // sqrt(3)

namespace cuda
{
    __device__ __forceinline__ float3 operator*(const Mat33& m, const float3& vec)
    {
        return make_float3(dot(m.data[0], vec), dot(m.data[1], vec), dot(m.data[2], vec));
    }

//    // calculates eigenvalues of 2x2 float symmetric matrix
//    __device__ void dsyevc2(float A, float B, float C, float* rt1, float* rt2) {
//        float sm = A + C;
//        float df = A - C;
//        float rt = sqrt(SQR(df) + 4.0 * B * B);
//        float t;
//
//        if (sm > 0.0)
//        {
//            *rt1 = 0.5 * (sm + rt);
//            t = 1.0 / (*rt1);
//            *rt2 = (A * t) * C - (B * t) * B;
//        }
//        else if (sm < 0.0)
//        {
//            *rt2 = 0.5 * (sm - rt);
//            t = 1.0 / (*rt2);
//            *rt1 = (A * t) * C - (B * t) * B;
//        }
//        else       // This case needs to be treated separately to avoid div by 0
//        {
//            *rt1 = 0.5 * rt;
//            *rt2 = -0.5 * rt;
//        }
//    }
//
//
//    // ----------------------------------------------------------------------------
//    // Calculates the eigensystem of a float symmetric 2x2 matrix
//    //    [ A  B ]
//    //    [ B  C ]
//    // in the form
//    //    [ A  B ]  =  [ cs  -sn ] [ rt1   0  ] [  cs  sn ]
//    //    [ B  C ]     [ sn   cs ] [  0   rt2 ] [ -sn  cs ]
//    // where rt1 >= rt2. Note that this convention is different from the one used
//    // in the LAPACK routine DLAEV2, where |rt1| >= |rt2|.
//    // ----------------------------------------------------------------------------
//    __device__ void dsyev2(float A, float B, float C, float* rt1, float* rt2, float* cs, float* sn)
//    {
//        float sm = A + C;
//        float df = A - C;
//        float rt = sqrt(SQR(df) + 4.0 * B * B);
//        float t;
//
//        if (sm > 0.0)
//        {
//            *rt1 = 0.5 * (sm + rt);
//            t = 1.0 / (*rt1);
//            *rt2 = (A * t) * C - (B * t) * B;
//        }
//        else if (sm < 0.0)
//        {
//            *rt2 = 0.5 * (sm - rt);
//            t = 1.0 / (*rt2);
//            *rt1 = (A * t) * C - (B * t) * B;
//        }
//        else       // This case needs to be treated separately to avoid div by 0
//        {
//            *rt1 = 0.5 * rt;
//            *rt2 = -0.5 * rt;
//        }
//
//        // Calculate eigenvectors
//        if (df > 0.0)
//            *cs = df + rt;
//        else
//            *cs = df - rt;
//
//        if (fabs(*cs) > 2.0 * fabs(B))
//        {
//            t = -2.0 * B / *cs;
//            *sn = 1.0 / sqrt(1.0 + SQR(t));
//            *cs = t * (*sn);
//        }
//        else if (fabs(B) == 0.0)
//        {
//            *cs = 1.0;
//            *sn = 0.0;
//        }
//        else
//        {
//            t = -0.5 * (*cs) / B;
//            *cs = 1.0 / sqrt(1.0 + SQR(t));
//            *sn = t * (*cs);
//        }
//
//        if (df > 0.0)
//        {
//            t = *cs;
//            *cs = -(*sn);
//            *sn = t;
//        }
//    }
//
//
//
//    // ----------------------------------------------------------------------------
//    // Calculates the eigenvalues of a symmetric 3x3 matrix A using Cardano's
//    // analytical algorithm.
//    // Only the diagonal and upper triangular parts of A are accessed. The access
//    // is read-only.
//    // ----------------------------------------------------------------------------
//    // Parameters:
//    //   A: The symmetric input matrix
//    //   w: Storage buffer for eigenvalues
//    // ----------------------------------------------------------------------------
//    // Return value:
//    //   0: Success
//    //  -1: Error
//    // ----------------------------------------------------------------------------
//    __device__ int dsyevc3(float A[3][3], float w[3])
//        
//    {
//        float m, c1, c0;
//
//        // Determine coefficients of characteristic poynomial. We write
//        //       | a   d   f  |
//        //  A =  | d*  b   e  |
//        //       | f*  e*  c  |
//        float de = A[0][1] * A[1][2];                                    // d * e
//        float dd = SQR(A[0][1]);                                         // d^2
//        float ee = SQR(A[1][2]);                                         // e^2
//        float ff = SQR(A[0][2]);                                         // f^2
//        m = A[0][0] + A[1][1] + A[2][2];
//        c1 = (A[0][0] * A[1][1] + A[0][0] * A[2][2] + A[1][1] * A[2][2])        // a*b + a*c + b*c - d^2 - e^2 - f^2
//            - (dd + ee + ff);
//        c0 = A[2][2] * dd + A[0][0] * ee + A[1][1] * ff - A[0][0] * A[1][1] * A[2][2]
//            - 2.0 * A[0][2] * de;                                     // c*d^2 + a*e^2 + b*f^2 - a*b*c - 2*f*d*e)
//
//        float p, sqrt_p, q, c, s, phi;
//        p = SQR(m) - 3.0 * c1;
//        q = m * (p - (3.0 / 2.0) * c1) - (27.0 / 2.0) * c0;
//        sqrt_p = sqrt(fabs(p));
//
//        phi = 27.0 * (0.25 * SQR(c1) * (p - c1) + c0 * (q + 27.0 / 4.0 * c0));
//        phi = (1.0 / 3.0) * atan2(sqrt(fabs(phi)), q);
//
//        c = sqrt_p * cos(phi);
//        s = (1.0 / M_SQRT3) * sqrt_p * sin(phi);
//
//        w[0] = (1.0 / 3.0) * (m - c);
//        w[1] = w[0] + s;
//        w[2] = w[0] + c;
//        w[0] -= s;
//
//        return 0;
//    }
//
//
//    // ----------------------------------------------------------------------------
//    __device__ int dsyevv3(float A[3][3], float Q[3][3], float w[3])
//        // ----------------------------------------------------------------------------
//        // Calculates the eigenvalues and normalized eigenvectors of a symmetric 3x3
//        // matrix A using Cardano's method for the eigenvalues and an analytical
//        // method based on vector cross products for the eigenvectors.
//        // Only the diagonal and upper triangular parts of A need to contain meaningful
//        // values. However, all of A may be used as temporary storage and may hence be
//        // destroyed.
//        // ----------------------------------------------------------------------------
//        // Parameters:
//        //   A: The symmetric input matrix
//        //   Q: Storage buffer for eigenvectors
//        //   w: Storage buffer for eigenvalues
//        // ----------------------------------------------------------------------------
//        // Return value:
//        //   0: Success
//        //  -1: Error
//        // ----------------------------------------------------------------------------
//        // Dependencies:
//        //   dsyevc3()
//        // ----------------------------------------------------------------------------
//        // Version history:
//        //   v1.1 (12 Mar 2012): Removed access to lower triangualr part of A
//        //     (according to the documentation, only the upper triangular part needs
//        //     to be filled)
//        //   v1.0: First released version
//        // ----------------------------------------------------------------------------
//    {
//#ifndef EVALS_ONLY
//        float norm;          // Squared norm or inverse norm of current eigenvector
//        float n0, n1;        // Norm of first and second columns of A
//        float n0tmp, n1tmp;  // "Templates" for the calculation of n0/n1 - saves a few FLOPS
//        float thresh;        // Small number used as threshold for floating point comparisons
//        float error;         // Estimated maximum roundoff error in some steps
//        float wmax;          // The eigenvalue of maximum modulus
//        float f, t;          // Intermediate storage
//        int i, j;             // Loop counters
//#endif
//
//  // Calculate eigenvalues
//        dsyevc3(A, w);
//
//#ifndef EVALS_ONLY
//        wmax = fabs(w[0]);
//        if ((t = fabs(w[1])) > wmax)
//            wmax = t;
//        if ((t = fabs(w[2])) > wmax)
//            wmax = t;
//        thresh = SQR(8.0 * DBL_EPSILON * wmax);
//
//        // Prepare calculation of eigenvectors
//        n0tmp = SQR(A[0][1]) + SQR(A[0][2]);
//        n1tmp = SQR(A[0][1]) + SQR(A[1][2]);
//        Q[0][1] = A[0][1] * A[1][2] - A[0][2] * A[1][1];
//        Q[1][1] = A[0][2] * A[0][1] - A[1][2] * A[0][0];
//        Q[2][1] = SQR(A[0][1]);
//
//        // Calculate first eigenvector by the formula
//        //   v[0] = (A - w[0]).e1 x (A - w[0]).e2
//        A[0][0] -= w[0];
//        A[1][1] -= w[0];
//        Q[0][0] = Q[0][1] + A[0][2] * w[0];
//        Q[1][0] = Q[1][1] + A[1][2] * w[0];
//        Q[2][0] = A[0][0] * A[1][1] - Q[2][1];
//        norm = SQR(Q[0][0]) + SQR(Q[1][0]) + SQR(Q[2][0]);
//        n0 = n0tmp + SQR(A[0][0]);
//        n1 = n1tmp + SQR(A[1][1]);
//        error = n0 * n1;
//
//        if (n0 <= thresh)         // If the first column is zero, then (1,0,0) is an eigenvector
//        {
//            Q[0][0] = 1.0;
//            Q[1][0] = 0.0;
//            Q[2][0] = 0.0;
//        }
//        else if (n1 <= thresh)    // If the second column is zero, then (0,1,0) is an eigenvector
//        {
//            Q[0][0] = 0.0;
//            Q[1][0] = 1.0;
//            Q[2][0] = 0.0;
//        }
//        else if (norm < SQR(64.0 * DBL_EPSILON) * error)
//        {                         // If angle between A[0] and A[1] is too small, don't use
//            t = SQR(A[0][1]);       // cross product, but calculate v ~ (1, -A0/A1, 0)
//            f = -A[0][0] / A[0][1];
//            if (SQR(A[1][1]) > t)
//            {
//                t = SQR(A[1][1]);
//                f = -A[0][1] / A[1][1];
//            }
//            if (SQR(A[1][2]) > t)
//                f = -A[0][2] / A[1][2];
//            norm = 1.0 / sqrt(1 + SQR(f));
//            Q[0][0] = norm;
//            Q[1][0] = f * norm;
//            Q[2][0] = 0.0;
//        }
//        else                      // This is the standard branch
//        {
//            norm = sqrt(1.0 / norm);
//            for (j = 0; j < 3; j++)
//                Q[j][0] = Q[j][0] * norm;
//        }
//
//
//        // Prepare calculation of second eigenvector
//        t = w[0] - w[1];
//        if (fabs(t) > 8.0 * DBL_EPSILON * wmax)
//        {
//            // For non-degenerate eigenvalue, calculate second eigenvector by the formula
//            //   v[1] = (A - w[1]).e1 x (A - w[1]).e2
//            A[0][0] += t;
//            A[1][1] += t;
//            Q[0][1] = Q[0][1] + A[0][2] * w[1];
//            Q[1][1] = Q[1][1] + A[1][2] * w[1];
//            Q[2][1] = A[0][0] * A[1][1] - Q[2][1];
//            norm = SQR(Q[0][1]) + SQR(Q[1][1]) + SQR(Q[2][1]);
//            n0 = n0tmp + SQR(A[0][0]);
//            n1 = n1tmp + SQR(A[1][1]);
//            error = n0 * n1;
//
//            if (n0 <= thresh)       // If the first column is zero, then (1,0,0) is an eigenvector
//            {
//                Q[0][1] = 1.0;
//                Q[1][1] = 0.0;
//                Q[2][1] = 0.0;
//            }
//            else if (n1 <= thresh)  // If the second column is zero, then (0,1,0) is an eigenvector
//            {
//                Q[0][1] = 0.0;
//                Q[1][1] = 1.0;
//                Q[2][1] = 0.0;
//            }
//            else if (norm < SQR(64.0 * DBL_EPSILON) * error)
//            {                       // If angle between A[0] and A[1] is too small, don't use
//                t = SQR(A[0][1]);     // cross product, but calculate v ~ (1, -A0/A1, 0)
//                f = -A[0][0] / A[0][1];
//                if (SQR(A[1][1]) > t)
//                {
//                    t = SQR(A[1][1]);
//                    f = -A[0][1] / A[1][1];
//                }
//                if (SQR(A[1][2]) > t)
//                    f = -A[0][2] / A[1][2];
//                norm = 1.0 / sqrt(1 + SQR(f));
//                Q[0][1] = norm;
//                Q[1][1] = f * norm;
//                Q[2][1] = 0.0;
//            }
//            else
//            {
//                norm = sqrt(1.0 / norm);
//                for (j = 0; j < 3; j++)
//                    Q[j][1] = Q[j][1] * norm;
//            }
//        }
//        else
//        {
//            // For degenerate eigenvalue, calculate second eigenvector according to
//            //   v[1] = v[0] x (A - w[1]).e[i]
//            //   
//            // This would floatly get to complicated if we could not assume all of A to
//            // contain meaningful values.
//            A[1][0] = A[0][1];
//            A[2][0] = A[0][2];
//            A[2][1] = A[1][2];
//            A[0][0] += w[0];
//            A[1][1] += w[0];
//            for (i = 0; i < 3; i++)
//            {
//                A[i][i] -= w[1];
//                n0 = SQR(A[0][i]) + SQR(A[1][i]) + SQR(A[2][i]);
//                if (n0 > thresh)
//                {
//                    Q[0][1] = Q[1][0] * A[2][i] - Q[2][0] * A[1][i];
//                    Q[1][1] = Q[2][0] * A[0][i] - Q[0][0] * A[2][i];
//                    Q[2][1] = Q[0][0] * A[1][i] - Q[1][0] * A[0][i];
//                    norm = SQR(Q[0][1]) + SQR(Q[1][1]) + SQR(Q[2][1]);
//                    if (norm > SQR(256.0 * DBL_EPSILON)* n0) // Accept cross product only if the angle between
//                    {                                         // the two vectors was not too small
//                        norm = sqrt(1.0 / norm);
//                        for (j = 0; j < 3; j++)
//                            Q[j][1] = Q[j][1] * norm;
//                        break;
//                    }
//                }
//            }
//
//            if (i == 3)    // This means that any vector orthogonal to v[0] is an EV.
//            {
//                for (j = 0; j < 3; j++)
//                    if (Q[j][0] != 0.0)                                   // Find nonzero element of v[0] ...
//                    {                                                     // ... and swap it with the next one
//                        norm = 1.0 / sqrt(SQR(Q[j][0]) + SQR(Q[(j + 1) % 3][0]));
//                        Q[j][1] = Q[(j + 1) % 3][0] * norm;
//                        Q[(j + 1) % 3][1] = -Q[j][0] * norm;
//                        Q[(j + 2) % 3][1] = 0.0;
//                        break;
//                    }
//            }
//        }
//
//
//        // Calculate third eigenvector according to
//        //   v[2] = v[0] x v[1]
//        Q[0][2] = Q[1][0] * Q[2][1] - Q[2][0] * Q[1][1];
//        Q[1][2] = Q[2][0] * Q[0][1] - Q[0][0] * Q[2][1];
//        Q[2][2] = Q[0][0] * Q[1][1] - Q[1][0] * Q[0][1];
//#endif
//
//        return 0;
//    }
//
//
//
//    // ----------------------------------------------------------------------------
//    __device__ void dsytrd3(float A[3][3], float Q[3][3], float d[3], float e[2])
//        // ----------------------------------------------------------------------------
//        // Reduces a symmetric 3x3 matrix to tridiagonal form by applying
//        // (unitary) Householder transformations:
//        //            [ d[0]  e[0]       ]
//        //    A = Q . [ e[0]  d[1]  e[1] ] . Q^T
//        //            [       e[1]  d[2] ]
//        // The function accesses only the diagonal and upper triangular parts of
//        // A. The access is read-only.
//        // ---------------------------------------------------------------------------
//    {
//        const int n = 3;
//        float u[n], q[n];
//        float omega, f;
//        float K, h, g;
//
//        // Initialize Q to the identitity matrix
//#ifndef EVALS_ONLY
//        for (int i = 0; i < n; i++)
//        {
//            Q[i][i] = 1.0;
//            for (int j = 0; j < i; j++)
//                Q[i][j] = Q[j][i] = 0.0;
//        }
//#endif
//
//        // Bring first row and column to the desired form 
//        h = SQR(A[0][1]) + SQR(A[0][2]);
//        if (A[0][1] > 0)
//            g = -sqrt(h);
//        else
//            g = sqrt(h);
//        e[0] = g;
//        f = g * A[0][1];
//        u[1] = A[0][1] - g;
//        u[2] = A[0][2];
//
//        omega = h - f;
//        if (omega > 0.0)
//        {
//            omega = 1.0 / omega;
//            K = 0.0;
//            for (int i = 1; i < n; i++)
//            {
//                f = A[1][i] * u[1] + A[i][2] * u[2];
//                q[i] = omega * f;                  // p
//                K += u[i] * f;                   // u* A u
//            }
//            K *= 0.5 * SQR(omega);
//
//            for (int i = 1; i < n; i++)
//                q[i] = q[i] - K * u[i];
//
//            d[0] = A[0][0];
//            d[1] = A[1][1] - 2.0 * q[1] * u[1];
//            d[2] = A[2][2] - 2.0 * q[2] * u[2];
//
//            // Store inverse Householder transformation in Q
//#ifndef EVALS_ONLY
//            for (int j = 1; j < n; j++)
//            {
//                f = omega * u[j];
//                for (int i = 1; i < n; i++)
//                    Q[i][j] = Q[i][j] - f * u[i];
//            }
//#endif
//
//            // Calculate updated A[1][2] and store it in e[1]
//            e[1] = A[1][2] - q[1] * u[2] - u[1] * q[2];
//        }
//        else
//        {
//            for (int i = 0; i < n; i++)
//                d[i] = A[i][i];
//            e[1] = A[1][2];
//        }
//    }
//
//    // ----------------------------------------------------------------------------
//    __device__ int dsyevq3(float A[3][3], float Q[3][3], float w[3])
//        // ----------------------------------------------------------------------------
//        // Calculates the eigenvalues and normalized eigenvectors of a symmetric 3x3
//        // matrix A using the QL algorithm with implicit shifts, preceded by a
//        // Householder reduction to tridiagonal form.
//        // The function accesses only the diagonal and upper triangular parts of A.
//        // The access is read-only.
//        // ----------------------------------------------------------------------------
//        // Parameters:
//        //   A: The symmetric input matrix
//        //   Q: Storage buffer for eigenvectors
//        //   w: Storage buffer for eigenvalues
//        // ----------------------------------------------------------------------------
//        // Return value:
//        //   0: Success
//        //  -1: Error (no convergence)
//        // ----------------------------------------------------------------------------
//        // Dependencies:
//        //   dsytrd3()
//        // ----------------------------------------------------------------------------
//    {
//        const int n = 3;
//        float e[3];                   // The third element is used only as temporary workspace
//        float g, r, p, f, b, s, c, t; // Intermediate storage
//        int nIter;
//        int m;
//
//        // Transform A to float tridiagonal form by the Householder method
//        dsytrd3(A, Q, w, e);
//
//        // Calculate eigensystem of the remaining float symmetric tridiagonal matrix
//        // with the QL method
//        //
//        // Loop over all off-diagonal elements
//        for (int l = 0; l < n - 1; l++)
//        {
//            nIter = 0;
//            while (1)
//            {
//                // Check for convergence and exit iteration loop if off-diagonal
//                // element e(l) is zero
//                for (m = l; m <= n - 2; m++)
//                {
//                    g = fabs(w[m]) + fabs(w[m + 1]);
//                    if (fabs(e[m]) + g == g)
//                        break;
//                }
//                if (m == l)
//                    break;
//
//                if (nIter++ >= 30)
//                    return -1;
//
//                // Calculate g = d_m - k
//                g = (w[l + 1] - w[l]) / (e[l] + e[l]);
//                r = sqrt(SQR(g) + 1.0);
//                if (g > 0)
//                    g = w[m] - w[l] + e[l] / (g + r);
//                else
//                    g = w[m] - w[l] + e[l] / (g - r);
//
//                s = c = 1.0;
//                p = 0.0;
//                for (int i = m - 1; i >= l; i--)
//                {
//                    f = s * e[i];
//                    b = c * e[i];
//                    if (fabs(f) > fabs(g))
//                    {
//                        c = g / f;
//                        r = sqrt(SQR(c) + 1.0);
//                        e[i + 1] = f * r;
//                        c *= (s = 1.0 / r);
//                    }
//                    else
//                    {
//                        s = f / g;
//                        r = sqrt(SQR(s) + 1.0);
//                        e[i + 1] = g * r;
//                        s *= (c = 1.0 / r);
//                    }
//
//                    g = w[i + 1] - p;
//                    r = (w[i] - g) * s + 2.0 * c * b;
//                    p = s * r;
//                    w[i + 1] = g + p;
//                    g = c * r - b;
//
//                    // Form eigenvectors
//#ifndef EVALS_ONLY
//                    for (int k = 0; k < n; k++)
//                    {
//                        t = Q[k][i + 1];
//                        Q[k][i + 1] = s * Q[k][i] + c * t;
//                        Q[k][i] = c * Q[k][i] - s * t;
//                    }
//#endif 
//                }
//                w[l] -= p;
//                e[l] = g;
//                e[m] = 0.0;
//            }
//        }
//
//        return 0;
//    }
//
//    /////// __global__ interface,column major for matlab
//    __device__ void eig2(const float* M, float* V, float* L) {
//        dsyev2(M[0], M[1], M[3], &L[3], &L[0], &V[1], &V[3]);
//        V[2] = V[1];
//        V[0] = -V[3];
//        L[1] = L[2] = 0;
//    }
//
//    __device__ void eig3(const float* M, float* V, float* L, bool useIterative = false) {
//
//        float A[3][3] = { {M[0],M[1],M[2]},
//                        {M[3],M[4],M[5]},
//                        {M[6],M[7],M[8]} };
//
//        float Q[3][3] = { {0,0,0},
//                        {0,0,0},
//                        {0,0,0} };
//
//        float LL[3] = { 0,0,0 };
//
//        int conv = 0;
//        if (useIterative) {
//            conv = dsyevq3(A, Q, LL);
//        }
//        else {
//            conv = dsyevv3(A, Q, LL);
//        }
//
//        if (conv < 0) {
//            L[0] = -1;
//            return;
//        }
//
//        L[0] = LL[0];
//        L[4] = LL[1];
//        L[8] = LL[2];
//
//        V[0] = Q[0][0]; V[1] = Q[1][0]; V[2] = Q[2][0];
//        V[3] = Q[0][1]; V[4] = Q[1][1]; V[5] = Q[2][1];
//        V[6] = Q[0][2]; V[7] = Q[1][2]; V[8] = Q[2][2];
//    }
//
//    __global__ void eig(const float* M, float* V, float* L, const int n, bool useIterative = false) {
//        if (n == 2) {
//            eig2(M, V, L);
//        }
//        else if (n == 3) {
//            eig3(M, V, L, useIterative);
//        }
//    }
//
//    __global__ void eigVal(const float* M, float* L, const int n) {
//        if (n == 2) {
//            dsyevc2(M[0], M[1], M[3], &L[1], &L[0]);
//        }
//        else if (n == 3) {
//            float A[3][3] = { {M[0],M[1],M[2]},
//                            {M[3],M[4],M[5]},
//                            {M[6],M[7],M[8]} };
//            dsyevc3(A, L);
//        }
//    }

}

namespace cuda
{
    struct ICP
    {
        IcpParameters parameters;
        float icpDistThreshold;
        float icpAnglesThreshold;
        pcl::gpu::PtrStepSz<ushort> srcDepthImage;
        pcl::gpu::PtrSz<float3> src;
        pcl::gpu::PtrSz<float3> dst;
        pcl::gpu::PtrSz<float3> srcNormals;
        pcl::gpu::PtrSz<float3> dstNormals;
        pcl::gpu::PtrSz<float3> rotSum;
        pcl::gpu::PtrSz<float3> transSum;
        pcl::gpu::PtrSz<float> cosSum;
        Mat33 initRot;
        float3 initTrans;

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
                printf("cx %f, cy %f, fx %f, fy %f, shift %f, width %d, height %d\n",
                    parameters.cx, parameters.cy, parameters.fx, parameters.fy, parameters.depthShift, parameters.depthWidth, parameters.depthHeight);
            }

            int ix = index % parameters.depthWidth;
            int iy = index / parameters.depthWidth;

            float x = 0, y = 0, z = 0;
            float zValue = srcDepthImage[index] * 1.f;
            z = zValue / parameters.depthShift;
            x = (ix - parameters.cx) * z / parameters.fx;
            y = (iy - parameters.cy) * z / parameters.fy;

            src[index].x = x;
            src[index].y = y;
            src[index].z = z;
        }

        //__device__ __forceinline__ void estimateNormals()
        //{
        //    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
        //    int ix = index % parameters.depthWidth;
        //    int iy = index / parameters.depthWidth;

        //    float3 center = make_float3(0, 0, 0);
        //    float C_2d[3][3] = { {0,0,0},{0,0,0},{0,0,0} };
        //    float* C = (float*)C_2d; //rowwise
        //    float3 normal;
        //    int count = 0;

        //    float3 point = src[index];
        //    if (!isValid(point))
        //        return;

        //    float maxDistance = parameters.normalKnnRadius * parameters.normalKnnRadius;

        //    for (int j = max(0, iy - parameters.normalKernalRadius); j < min(iy + parameters.normalKernalRadius + 1, parameters.depthHeight); j++) {
        //        for (int i = max(0, ix - parameters.normalKernalRadius); i < min(ix + parameters.normalKernalRadius + 1, parameters.depthWidth); i++) {
        //            size_t neighbourIndex = j * parameters.depthWidth + i;
        //            float3 diff3 = src[neighbourIndex] - src[index];
        //            float norm = dot(diff3, diff3);
        //            if (norm < maxDistance) {
        //                center += src[neighbourIndex];
        //                outerAdd(src[neighbourIndex], C); //note: instead of pts[ind]-center, we demean afterwards
        //                count++;
        //            }
        //        }
        //    }
        //    if (count < 3)
        //        return;

        //    center /= (count + 0.0f);
        //    outerAdd(center, C, -count);
        //    float fac = 1.0f / (count - 1.0f);
        //    mul(C, fac);
        //    float Q[3][3] = { {0,0,0},{0,0,0},{0,0,0} };
        //    float w[3] = { 0, 0, 0 };
        //    int result = dsyevv3(C_2d, Q, w);

        //    normal.x = Q[0][0];
        //    normal.y = Q[1][0];
        //    normal.z = Q[2][0];

        //    //printf("index: %d, ix: %d, iy: %d, x: %f, y: %f, z: %f, depth: %d\n", index, ix, iy, normal.x, normal.y, normal.z, count);

        //    srcNormals[index].x = normal.x;
        //    srcNormals[index].y = normal.y;
        //    srcNormals[index].z = normal.z;
        //}

        __device__ __forceinline__ void step()
        {
            size_t index = blockIdx.x * blockDim.x + threadIdx.x;
            int ix = index % parameters.depthWidth;
            int iy = index / parameters.depthWidth;

            float3 ptSrc = src[index];
            float3 nmSrc = srcNormals[index];
            float3 ptDst = dst[index];
            float3 nmDst = dstNormals[index];

            ptSrc = initRot * ptSrc + initTrans;
            float maxDistance = parameters.icpDistThreshold * parameters.icpDistThreshold;

            bool found = false;
            float maxCos = 0;
            float3 foundDstPt;
            float3 foundDstNm;
            for (int j = max(0, iy - parameters.normalKernalRadius); j < min(iy + parameters.normalKernalRadius + 1, parameters.depthHeight); j++) {
                for (int i = max(0, ix - parameters.normalKernalRadius); i < min(ix + parameters.normalKernalRadius + 1, parameters.depthWidth); i++) {
                    size_t dstIndex = j * parameters.depthWidth + i;
                    float3 diff3 = dst[dstIndex] - src[index];
                    float distSqr = dot(diff3, diff3);
                    //printf("index: %d, ix: %d, iy: %d, norm: %f, max distance: %f\n", index, ix, iy, norm, maxDistance);
                    if (distSqr < maxDistance) {
                        float cos = dot(nmSrc, dstNormals[dstIndex]);
                        if (cos < parameters.icpAnglesThreshold)
                            continue;

                        if (cos > maxCos)
                        {
                            maxCos = cos;
                            foundDstPt = dst[dstIndex];
                            foundDstNm = dstNormals[dstIndex];
                            found = true;
                        }
                    }
                }
            }

            if (found)
            {
                rotSum[index] = cross(ptSrc, foundDstPt);
                transSum[index] = foundDstNm;
                cosSum[index] = dot(foundDstNm, (foundDstPt - ptSrc));
            }
        }
    };

    void IcpGenerateCloud(IcpFrame& frame)
    {
        dim3 block(16);
        dim3 grid(frame.parameters.depthWidth * frame.parameters.depthHeight / block.x);

        int size = frame.parameters.depthWidth * frame.parameters.depthHeight;
        cudaMemset(frame.pointCloud.ptr(), 0, size * sizeof(float3));
        cudaMemset(frame.pointCloudCache.ptr(), 0, size * sizeof(float3));
        cudaMemset(frame.pointCloudNormals.ptr(), 0, size * sizeof(float3));
        safeCall(cudaDeviceSynchronize());

        ICP icp;
        icp.parameters = frame.parameters;


    }

    void icp(IcpFrame& frame1, IcpFrame& frame2, const Mat33& initRot, const float3& initTrans, Mat33& rot, float3& trans, float& error)
    {

    }
    
}