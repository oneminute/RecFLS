#ifndef LINEMATCHCUDAODOMETRY_H
#define LINEMATCHCUDAODOMETRY_H

#include <QObject>

#include "Odometry.h"
#include "cuda/CudaInternal.h"

#include <pcl/common/common.h>
#include <pcl/gpu/containers/device_array.h>
#include <cuda_runtime.h>

class LineMatchCudaOdometry : public Odometry
{
    Q_OBJECT
public:
    explicit LineMatchCudaOdometry(
        int bilateralFilterKernelSize = 5, 
        float bilateralSigmaColor = 100, 
        float bilateralSigmaSpatial = 100, 
        int normalEstimationKernelHalfSize = 9,
        float normalEstimationMaxDistance = 0.05f,
        QObject* parent = nullptr)
        : Odometry(parent)
        , m_bilateralFilterKernelSize(bilateralFilterKernelSize)
        , m_bilateralFilterSigmaColor(bilateralSigmaColor)
        , m_bilateralFilterSigmaSpatial(bilateralSigmaSpatial)
        , m_normalEstimationKernelHalfSize(normalEstimationKernelHalfSize)
        , m_normalEstimationMaxDistance(normalEstimationMaxDistance)
        , m_init(false)
    {}

    // Inherited via Odometry
    virtual void doProcessing(Frame& frame) override;
    virtual void afterProcessing(Frame& frame) override;
    virtual bool beforeProcessing(Frame& frame);
    virtual void saveCurrentFrame() override;

private:
    cv::cuda::GpuMat m_colorMatGpu;
    cv::cuda::GpuMat m_depthMatGpu;
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_boundaryCloud;

    cuda::GpuFrame m_frameGpu;

    bool m_init;

    int m_bilateralFilterKernelSize;
    float m_bilateralFilterSigmaColor;
    float m_bilateralFilterSigmaSpatial;
    float m_normalEstimationMaxDistance;
    float m_normalEstimationKernelHalfSize;
};

#endif // LINEMATCHCUDAODOMETRY_H
