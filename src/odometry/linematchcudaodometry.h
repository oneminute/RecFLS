#ifndef LINEMATCHCUDAODOMETRY_H
#define LINEMATCHCUDAODOMETRY_H

#include <QObject>

#include "Odometry.h"

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
};

#endif // LINEMATCHCUDAODOMETRY_H
