#include "linematchcudaodometry.h"
#include "util/StopWatch.h"

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/cudaimgproc.hpp>

LineMatchCudaOdometry::LineMatchCudaOdometry(QObject *parent)
    : Odometry(parent)
    , m_colorMatGpu(480, 640, CV_8UC3)
    , m_depthMatGpu(480, 640, CV_16U)
{
}

bool LineMatchCudaOdometry::beforeProcessing(Frame& frame)
{
    return true;
}

void LineMatchCudaOdometry::doProcessing(Frame& frame)
{
    TICK("odometry_uploading");
    m_colorMatGpu.upload(frame.colorMat());
    m_depthMatGpu.upload(frame.depthMat());
    TOCK("odometry_uploading");

    TICK("odometry_bilateral_filter");
    cv::cuda::bilateralFilter(m_depthMatGpu, m_depthMatGpu, 5, 100, 100);
    TOCK("odometry_bilateral_filter");

    TICK("odometry_downloading");
    cv::Mat depthMatCpu;
    m_depthMatGpu.download(depthMatCpu);
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
