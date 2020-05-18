#ifndef LINEMATCHCUDAODOMETRY_H
#define LINEMATCHCUDAODOMETRY_H

#include <QObject>

#include "Odometry.h"
#include "cuda/CudaInternal.h"

#include <pcl/common/common.h>
#include <pcl/gpu/containers/device_array.h>
#include <cuda_runtime.h>

#include "extractor/BoundaryExtractor.h"
#include "extractor/FusedLineExtractor.h"
#include "matcher/LineMatcher.h"
#include "device/SensorReaderDevice.h"
#include "extractor/LineSegment.h"

class LineMatchCudaOdometry : public Odometry
{
    Q_OBJECT
public:
    explicit LineMatchCudaOdometry(
        QObject* parent = nullptr)
        : Odometry(parent)
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

    QScopedPointer<BoundaryExtractor> m_boundaryExtractor;
    QScopedPointer<FusedLineExtractor> m_lineExtractor;
    QScopedPointer<LineMatcher> m_lineMatcher;

    QList<pcl::PointCloud<LineSegment>::Ptr> m_lines;
};

#endif // LINEMATCHCUDAODOMETRY_H
