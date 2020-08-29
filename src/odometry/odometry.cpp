#include "Odometry.h"
#include "util/StopWatch.h"

Odometry::Odometry(QObject *parent) : QObject(parent)
    , m_cloud(new pcl::PointCloud<pcl::PointXYZRGB>)
{

}

void Odometry::process(Frame& frame)
{
    TICK("odometry_process");
    // һ�����͵Ľӿڵ��ã��ڵ���ʵ�����doProcessing����֮ǰ����ִ��beforeProcessing��֮��ִ��afterProcessing���������û�������ǰ���º��������
    m_filteredMats.clear();
    if (!beforeProcessing(frame))
    {
        return;
    }

    // ʵ�����doProcessing�������ã�ִ��ʵ��ҵ����롣
    doProcessing(frame);
    //frame.setFrameIndex(m_frames.size());

    afterProcessing(frame);
    TOCK("odometry_process");
}

void Odometry::setCloudViewer(CloudViewer* viewer)
{
    m_cloudViewer = viewer;
}
