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
    explicit LineMatchCudaOdometry(QObject* parent = nullptr);

    // Inherited via Odometry
    virtual void doProcessing(Frame& frame) override;
    virtual void afterProcessing(Frame& frame) override;
    virtual bool beforeProcessing(Frame& frame);

private:
    cv::cuda::GpuMat m_colorMatGpu;
    cv::cuda::GpuMat m_depthMatGpu;
    pcl::gpu::DeviceArray<float4> m_pointCloudGpu;
    pcl::gpu::DeviceArray2D<uchar3> m_colorBuffer;
    pcl::gpu::DeviceArray2D<ushort> m_depthBuffer;

    cuda::Parameters m_parameters;
    cuda::Frame m_frameGpu;

    bool m_init;
};

#endif // LINEMATCHCUDAODOMETRY_H
