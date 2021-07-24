#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <QObject>
#include <QScopedPointer>

#include "common/Frame.h"
#include "device/Device.h"
#include "ui/CloudViewer.h"
#include "odometry/Odometry.h"

#include <pcl/common/common.h>

class Controller : public QObject
{
    Q_OBJECT
public:
    explicit Controller(Device *device, QObject *parent = nullptr);

    virtual QString name() const = 0;

    bool supportRandomAccessing() const;

    virtual bool open() = 0;

    virtual void close() = 0;

    virtual void fetchNext() = 0;

	virtual void start() = 0;

    virtual void moveTo(int frameIndex) = 0;

    virtual void skip(int frameNumbers) = 0;

    virtual void reset() = 0;

    virtual Frame getFrame(int frameIndex) = 0;

    virtual void saveCurrentFrame() {}

    QList<QPair<QString, cv::Mat>>& filteredMats() {
        return m_odometry->filteredMats();
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud() {
        return m_odometry->cloud();
    }

    pcl::PointCloud<pcl::Normal>::Ptr normals() {
        return m_odometry->normals();
    }

    pcl::IndicesPtr cloudIndices() {
        return m_odometry->cloudIndices();
    }

    void setCloudViewer(CloudViewer *cloudViewer)
    {
        m_cloudViewer = cloudViewer;
    }

    Eigen::Matrix4f pose() const
    {
        return m_odometry->pose();
    }

    QMap<qint64, Eigen::Matrix4f> poses() const
    {
        return m_odometry->poses();
    }

signals:
    void frameFetched(Frame& frame);

public slots:

protected:
    Device *m_device;
    CloudViewer *m_cloudViewer;
    QScopedPointer<Odometry> m_odometry;
};

#endif // CONTROLLER_H
