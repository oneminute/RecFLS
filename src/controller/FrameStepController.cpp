#include "FrameStepController.h"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"

FrameStepController::FrameStepController(Device *device, QObject *parent)
    : Controller(device, parent)
{
    connect(m_device, &Device::frameFetched, this, &FrameStepController::onFrameFetched);
}

QString FrameStepController::name() const
{
    return "FrameStepController";
}

bool FrameStepController::open()
{
    return m_device->open();
}

void FrameStepController::close()
{
    m_device->close();
}

void FrameStepController::fetchNext()
{
    m_device->fetchNext();
}

void FrameStepController::moveTo(int frameIndex)
{
}

void FrameStepController::skip(int frameNumbers)
{
}

void FrameStepController::reset()
{
}

Frame FrameStepController::getFrame(int frameIndex)
{
    Frame frame;
    return frame;
}

void FrameStepController::onFrameFetched(Frame &frame)
{
    // rectify image

    // align depth image to color image

    // gaussian filter

    // bilateral filter

    // generate organized point cloud

    // boundary estimation and extract lines

    // generate line descriptors

    // match

    // calculate transformation

    // emit signal
    emit frameFetched(frame);
}
