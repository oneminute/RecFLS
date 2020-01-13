#ifndef ODOMETRY_H
#define ODOMETRY_H

#include <QObject>

#include "common/Frame.h"
#include "ui/CloudViewer.h"

class Odometry : public QObject
{
    Q_OBJECT
public:
    explicit Odometry(QObject *parent = nullptr);

    void process(Frame& frame);

    void setCloudViewer(CloudViewer* viewer);

    QList<QPair<QString, cv::Mat>>& filteredMats() {
        return m_filteredMats;
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud() {
        return m_cloud;
    }

    pcl::PointCloud<pcl::Normal>::Ptr normals() {
        return m_normals;
    }

    pcl::IndicesPtr cloudIndices() {
        return m_cloudIndices;
    }

    virtual void saveCurrentFrame() {}

signals:

protected:
    virtual bool beforeProcessing(Frame& frame) = 0;
    virtual void doProcessing(Frame& frame) = 0;
    virtual void afterProcessing(Frame& frame) = 0;

protected:
    CloudViewer *m_cloudViewer;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr m_cloud;
    pcl::PointCloud<pcl::Normal>::Ptr m_normals;
    pcl::IndicesPtr m_cloudIndices;
    QList<QPair<QString, cv::Mat>> m_filteredMats;
};

#endif // ODOMETRY_H
