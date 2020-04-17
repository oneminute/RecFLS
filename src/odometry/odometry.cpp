#include "Odometry.h"
#include "util/StopWatch.h"

Odometry::Odometry(QObject *parent) : QObject(parent)
    , m_cloud(new pcl::PointCloud<pcl::PointXYZRGB>)
{

}

void Odometry::process(Frame& frame)
{
    TICK("odometry_process");
    // 一个典型的接口调用，在调用实例类的doProcessing函数之前，先执行beforeProcessing，之后执行afterProcessing函数，给用户处理事前与事后的能力。
    m_filteredMats.clear();
    if (!beforeProcessing(frame))
    {
        return;
    }

    // 实例类的doProcessing函数调用，执行实际业务代码。
    doProcessing(frame);
    //frame.setFrameIndex(m_frames.size());

    afterProcessing(frame);
    TOCK("odometry_process");
}

void Odometry::setCloudViewer(CloudViewer* viewer)
{
    m_cloudViewer = viewer;
}
