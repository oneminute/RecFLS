#include "linematchcudaodometry.h"
#include "util/StopWatch.h"

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/cudaimgproc.hpp>

LineMatchCudaOdometry::LineMatchCudaOdometry(QObject *parent)
    : Odometry(parent)
{

}

bool LineMatchCudaOdometry::beforeProcessing(Frame& frame)
{
    return true;
}

void LineMatchCudaOdometry::doProcessing(Frame& frame)
{
    TICK("odometry_uploading");
    cv::cuda::GpuMat colorMatGpu(frame.colorMat());
    cv::cuda::GpuMat depthMatGpu(frame.depthMat());
    TOCK("odometry_uploading");

    TICK("odometry_bilateral_filter");
    cv::cuda::bilateralFilter(depthMatGpu, depthMatGpu, 5, 100, 100);
    TOCK("odometry_bilateral_filter");

    TICK("odometry_downloading");
    cv::Mat depthMatCpu;
    depthMatGpu.download(depthMatCpu);
    TOCK("odometry_downloading");

    cv::Mat colorMat = frame.colorMat();
    cv::Mat rectifiedColorMat = frame.undistortRGBImage();
    m_filteredMats.append(QPair<QString, cv::Mat>("rectified color image", rectifiedColorMat.clone()));

    m_filteredMats.append(QPair<QString, cv::Mat>("bilateral depth image", depthMatCpu.clone()));
    cv::Mat diff = frame.depthMat() - depthMatCpu;
    m_filteredMats.append(QPair<QString, cv::Mat>("diff depth image", diff.clone()));
}

void LineMatchCudaOdometry::afterProcessing(Frame& frame)
{
}
