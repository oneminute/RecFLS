#include "linematchcudaodometry.h"

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
//#include <opencv2/cudaimgproc.hpp>

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
