#ifndef ODOMETRY_H
#define ODOMETRY_H

#include <QObject>

#include "common/Frame.h"
#include "common/FLFrame.h"
#include "common/RelInformation.h"
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

    Eigen::Matrix4f pose() const
    {
        return m_pose;
    }

    Eigen::Matrix4f pose(qint64 index)
    {
        if (m_poses.contains(index))
            return m_poses[index];
        else
            return Eigen::Matrix4f::Identity();
    }

    QMap<qint64, Eigen::Matrix4f> poses() const
    {
        return m_poses;
    }

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
    QList<Frame> m_frames;
    QMap<qint64, Eigen::Matrix4f> m_poses;
    //QList<Eigen::Matrix4f> m_relPoses;
    QMap<RelInformation::KeyPair, RelInformation> m_relInfors;

    Eigen::Matrix4f m_pose;
};

#endif // ODOMETRY_H
