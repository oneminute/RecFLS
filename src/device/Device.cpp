#include "Device.h"
#include "util/Utils.h"
#include "common/Parameters.h"
#include "device/IclNuimDevice.h"
#include "device/SensorReaderDevice.h"

#include <pcl/point_types.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <QDebug>

Device::Device(QObject *parent)
    : QObject(parent)
    , m_colorIntrinsic(Eigen::Matrix4f::Zero())
    , m_colorExtrinsic(Eigen::Matrix4f::Zero())
    , m_depthIntrinsic(Eigen::Matrix4f::Zero())
    , m_depthExtrinsic(Eigen::Matrix4f::Zero())
    , m_R(Eigen::Matrix3f::Zero())
    , m_colorSize(0, 0)
    , m_depthSize(0, 0)
    , m_depthShift(0)
{

}

void Device::initRectifyMap()
{
    cv::Mat cameraMatrix = cvMatFrom(m_colorIntrinsic.topLeftCorner(3, 3));
    cv::Mat distCoeffs;
    if (m_distCoeffs.size() < 4)
    {
        distCoeffs = cv::Mat(1, 4, CV_32F, cv::Scalar(0));
    }
    else
    {
        distCoeffs = cvMatFrom(m_distCoeffs);
    }
    cv::Size imgSize(m_colorSize.width(), m_colorSize.height());
    cv::Mat newCameraMatrix = cv::getOptimalNewCameraMatrix(cameraMatrix, distCoeffs, imgSize, 1, imgSize, 0);
    cv::Mat R;
    if (!m_R.isZero())
    {
        R = cvMatFrom(m_R);
    }
    cv::initUndistortRectifyMap(cameraMatrix, distCoeffs, R, newCameraMatrix, imgSize, CV_32F, m_rectifyMap1, m_rectifyMap2);
    //qDebug() << cameraMatrix;
    //qDebug() << distCoeffs;
    //qDebug() << newCameraMatrix;
    //qDebug() << R;
    //qDebug() << m_rectifyMap1;
    //qDebug() << m_rectifyMap2;
}

cv::Mat Device::undistortImage(const cv::Mat &in)
{
    cv::Mat out;
    cv::remap(in, out, m_rectifyMap1, m_rectifyMap2, cv::INTER_LINEAR);
    return out;
}

cv::Mat Device::alignDepthToColor(const cv::Mat &depthMat, const cv::Mat &colorMat)
{
    cv::Mat out;
    for (int r = 0; r < depthMat.rows; r++)
    {
        for (int c = 0; c < depthMat.cols; c++)
        {

        }
    }
    return out;
}

float Device::fx() const
{
    return m_colorIntrinsic(0, 0);
}

float Device::fy() const
{
    return m_colorIntrinsic(1, 1);
}

float Device::cx() const
{
    return m_colorIntrinsic(2, 0);
}

float Device::cy() const
{
    return m_colorIntrinsic(2, 1);
}

float Device::depthShift() const
{
    return m_depthShift;
}

Device* Device::createDevice()
{
    Device* device = nullptr;
    if (Settings::Device_DeviceName.value() == "SensorReader")
    {
        device = new SensorReaderDevice;
    }
    else if (Settings::Device_DeviceName.value() == "IclNuim")
    {
        device = new IclNuimDevice;
    }
    return device;
}
