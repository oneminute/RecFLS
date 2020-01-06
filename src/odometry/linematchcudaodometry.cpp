#include "linematchcudaodometry.h"
#include "util/StopWatch.h"
#include "cuda/CudaInternal.h"
#include "device/Device.h"

#include <QDebug>

#include <pcl/common/common.h>
#include <pcl/gpu/containers/device_array.h>
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/cudaimgproc.hpp>

LineMatchCudaOdometry::LineMatchCudaOdometry(QObject *parent)
    : Odometry(parent)
    , m_init(false)
{

}

bool LineMatchCudaOdometry::beforeProcessing(Frame& frame)
{
    if (!m_init)
    {
        m_init = true;
        m_colorBuffer.create(frame.getColorHeight(), frame.getColorWidth());
        m_depthBuffer.create(frame.getDepthHeight(), frame.getDepthWidth());
        m_pointCloudGpu.create(frame.getDepthWidth() * frame.getDepthHeight());

        m_colorMatGpu = cv::cuda::GpuMat(frame.getColorHeight(), frame.getColorWidth(), CV_8UC3, m_colorBuffer);
        m_depthMatGpu = cv::cuda::GpuMat(frame.getDepthHeight(), frame.getDepthWidth(), CV_16U, m_depthBuffer);

        m_parameters.cx = frame.getDevice()->cx();
        m_parameters.cy = frame.getDevice()->cy();
        m_parameters.fx = frame.getDevice()->fx();
        m_parameters.fy = frame.getDevice()->fy();
        m_parameters.colorWidth = frame.getColorWidth();
        m_parameters.colorHeight = frame.getColorHeight();
        m_parameters.depthWidth = frame.getDepthWidth();
        m_parameters.depthHeight = frame.getDepthHeight();
        m_parameters.depthShift = frame.getDevice()->depthShift();

        m_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
        m_cloudIndices.reset(new std::vector<int>);
    }
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
    
    m_frameGpu.colorImage = m_colorBuffer;
    m_frameGpu.depthImage = m_depthBuffer;
    m_frameGpu.pointCloud = m_pointCloudGpu;

    cuda::generatePointCloud(m_parameters, m_frameGpu);

    TICK("odometry_downloading");
    cv::Mat depthMatCpu;
    m_depthMatGpu.download(depthMatCpu);
    std::vector<float4> points;
    m_frameGpu.pointCloud.download(points);
    TOCK("odometry_downloading");

    for(int i = 0; i < frame.getDepthHeight(); i++) {
        for(int j = 0; j < frame.getDepthWidth(); j++) {
            int index = i * frame.getDepthWidth() + j;
            float4 value = points[index];
            pcl::PointXYZRGBNormal pt;

            pt.x = value.x;
            pt.y = value.y;
            pt.z = value.z;
            pt.b = frame.colorMat().at<cv::Vec3b>(cv::Point(j, i))[0];
            pt.g = frame.colorMat().at<cv::Vec3b>(cv::Point(j, i))[1];
            pt.r = frame.colorMat().at<cv::Vec3b>(cv::Point(j, i))[2];

            if (pt.getVector3fMap().norm() > 0.1f) {
                m_cloudIndices->push_back(index);
                m_cloud->push_back(pt);
                //qDebug() << pt.x << pt.y << pt.z;
            }
            else
            {
                pt.x = qQNaN();
                pt.y = qQNaN();
                pt.z = qQNaN();
            }

        }
    }

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
