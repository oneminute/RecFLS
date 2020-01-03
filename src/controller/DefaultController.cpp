#include "DefaultController.h"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"

#include "ui/CloudViewer.h"

#include <QDebug>
#include <QDateTime>

#include "odometry/LineMatchOdometry.h"

DefaultController::DefaultController(Device *device, QObject *parent)
    : Controller(device, parent)
{
    connect(m_device, &Device::frameFetched, this, &DefaultController::onFrameFetched);

    m_odometry.reset(new LineMatchOdometry);
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

void DefaultController::onFrameFetched(Frame &frame)
{
    m_odometry->setCloudViewer(m_cloudViewer);
    m_odometry->process(frame);

    // emit signal
    emit frameFetched(frame);
}
