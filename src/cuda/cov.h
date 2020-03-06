#include "cuda_runtime.h"

__device__ float3 d_mean(const float3* pts, const int n);
__device__ void d_cov(const float3* pts, float* C, const int n);
__global__ void mean(const float* pts, float* m, const int n);
__global__ void cov(const float* pts, float* C, const int n);

//namespace cuda
//{
//#define SQR(x)      ((x)*(x))                        // x^2 
//#define SQR_ABS(x)  (SQR(cfloat(x)) + SQR(cimag(x)))  // |x|^2
//
//#define M_SQRT3    1.73205080756887729352744634151   // sqrt(3)
//
//
//    // calculates eigenvalues of 2x2 float symmetric matrix
//    __device__ void dsyevc2(float A, float B, float C, float* rt1, float* rt2);
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
//    __device__ void dsyev2(float A, float B, float C, float* rt1, float* rt2, float* cs, float* sn);
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
//    __device__ int dsyevc3(float A[3][3], float w[3]);
//
//    // ----------------------------------------------------------------------------
//    __device__ int dsyevv3(float A[3][3], float Q[3][3], float w[3]);
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
//
//    // ----------------------------------------------------------------------------
//    __device__ void dsytrd3(float A[3][3], float Q[3][3], float d[3], float e[2]);
//        // ----------------------------------------------------------------------------
//        // Reduces a symmetric 3x3 matrix to tridiagonal form by applying
//        // (unitary) Householder transformations:
//        //            [ d[0]  e[0]       ]
//        //    A = Q . [ e[0]  d[1]  e[1] ] . Q^T
//        //            [       e[1]  d[2] ]
//        // The function accesses only the diagonal and upper triangular parts of
//        // A. The access is read-only.
//        // ---------------------------------------------------------------------------
//
//    // ----------------------------------------------------------------------------
//    __device__ int dsyevq3(float A[3][3], float Q[3][3], float w[3]);
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
//
//    /////// __global__ interface,column major for matlab
//    __device__ void eig2(const float* M, float* V, float* L); 
//
//    __device__ void eig3(const float* M, float* V, float* L, bool useIterative = false); 
//
//    __global__ void eig(const float* M, float* V, float* L, const int n, bool useIterative = false); 
//
//    __global__ void eigVal(const float* M, float* L, const int n); 
//
//}
