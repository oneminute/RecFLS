#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <QObject>

#include "common/Frame.h"
#include "device/Device.h"
#include "ui/CloudViewer.h"

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

    virtual void moveTo(int frameIndex) = 0;

    virtual void skip(int frameNumbers) = 0;

    virtual void reset() = 0;

    virtual Frame getFrame(int frameIndex) = 0;

    QList<QPair<QString, cv::Mat>>& filteredMats() {
        return m_filteredMats;
    }

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud() {
        return m_cloud;
    }

    pcl::IndicesPtr cloudIndices() {
        return m_cloudIndices;
    }

    void setCloudViewer(CloudViewer *cloudViewer)
    {
        m_cloudViewer = cloudViewer;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr result()
    {
        return m_result;
    }

signals:
    void frameFetched(Frame& frame);

public slots:

protected:
    Device *m_device;
    QList<QPair<QString, cv::Mat>> m_filteredMats;
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr m_cloud;
    pcl::IndicesPtr m_cloudIndices;
    CloudViewer *m_cloudViewer;
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_result;
};

#endif // CONTROLLER_H
