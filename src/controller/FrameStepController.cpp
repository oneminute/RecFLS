#include "FrameStepController.h"

FrameStepController::FrameStepController()
{

}

QString FrameStepController::name() const
{
    return "FrameStepController";
}


bool FrameStepController::supportRandomAccessing() const
{
    return true;
}

void FrameStepController::fetchNext()
{
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
