#include "ICPMatcher.h"

#include <QObject>

ICPMatcher::ICPMatcher(QObject* parent)
    : QObject(parent)
    , PROPERTY_INIT(MaxIterations, 10)
{
    
}

Eigen::Matrix4f ICPMatcher::compute(cuda::GpuFrame* frame1, cuda::GpuFrame* frame2, const Eigen::Matrix4f& initPose, float& error)
{
    return Eigen::Matrix4f();
}
