#include "DefaultController.h"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"

#include "ui/CloudViewer.h"

#include <QDebug>
#include <QDateTime>

#include "odometry/LineMatchOdometry.h"
#include "odometry/LineMatchCudaOdometry.h"
#include "common/Parameters.h"

DefaultController::DefaultController(Device *device, QObject *parent)
    : Controller(device, parent)
{
    connect(m_device, &Device::frameFetched, this, &DefaultController::onFrameFetched);

    m_odometry.reset(new LineMatchCudaOdometry(
        Parameters::Global().intValue("bilateral_filter_kernel_size", 5, "LineMatchOdometry"),
        Parameters::Global().floatValue("bilateral_filter_sigma_color", 100, "LineMatchOdometry"),
        Parameters::Global().floatValue("bilateral_filter_sigma_spatial", 100, "LineMatchOdometry"),
        Parameters::Global().intValue("normal_estimation_kernel_half_size", 9, "LineMatchOdometry"),
        Parameters::Global().floatValue("normal_estimation_max_distance", 0.05f, "LineMatchOdometry")));
    //m_odometry.reset(new LineMatchOdometry);
}

QString DefaultController::name() const
{
    return "DefaultController";
}

bool DefaultController::open()
{
    return m_device->open();
}

void DefaultController::close()
{
    m_device->close();
}

void DefaultController::fetchNext()
{
    m_device->fetchNext();
}

void DefaultController::moveTo(int frameIndex)
{
}

void DefaultController::skip(int frameNumbers)
{
}

void DefaultController::reset()
{
}

Frame DefaultController::getFrame(int frameIndex)
{
    Frame frame;
    return frame;
}

void DefaultController::saveCurrentFrame()
{
    m_odometry->saveCurrentFrame();
}

void DefaultController::onFrameFetched(Frame &frame)
{
    // 处理来自设备的数据帧
    m_odometry->setCloudViewer(m_cloudViewer);

    // 里程计处理单帧数据
    m_odometry->process(frame);

    // emit signal
    emit frameFetched(frame);
}
