#ifndef ICPMATCHER_H
#define ICPMATCHER_H

#include <QObject>

#include <pcl/common/common.h>
#include <pcl/features/eigen.h>
#include <pcl/search/kdtree.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <extractor/LineExtractor.h>
#include <cuda/CudaInternal.h>

class ICPMatcher : public QObject
{
    Q_OBJECT
public:
    explicit ICPMatcher(QObject* parent = nullptr);

    Eigen::Matrix4f compute(cuda::GpuFrame* frame1, cuda::GpuFrame* frame2, const Eigen::Matrix4f& initPose, float& error);

protected:

private:
    

    PROPERTY(int, MaxIterations)
};

#endif // ICPMATCHER_H