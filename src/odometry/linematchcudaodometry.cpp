#include "linematchcudaodometry.h"

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
}

void LineMatchCudaOdometry::afterProcessing(Frame& frame)
{
}
