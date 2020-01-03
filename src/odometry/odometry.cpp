#include "odometry.h"
#include "util/StopWatch.h"

Odometry::Odometry(QObject *parent) : QObject(parent)
{

}

void Odometry::process(Frame& frame)
{
    TICK("odometry_process");
    m_filteredMats.clear();
    if (!beforeProcessing(frame))
    {
        return;
    }

    doProcessing(frame);

    afterProcessing(frame);
    TOCK("odometry_process");
}

void Odometry::setCloudViewer(CloudViewer* viewer)
{
    m_cloudViewer = viewer;
}
