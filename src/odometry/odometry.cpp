#include "odometry.h"

Odometry::Odometry(QObject *parent) : QObject(parent)
{

}

void Odometry::process(Frame& frame)
{
    if (!beforeProcessing(frame))
    {
        return;
    }

    doProcessing(frame);

    afterProcessing(frame);
}
