#ifndef FLFRAME_H
#define FLFRAME_H

#include <QObject>
#include <QSharedDataPointer>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <extractor/LineSegment.h>
#include <pcl/common/common.h>

class FLFrameData;

class FLFrame : public QObject
{
    Q_OBJECT
public:
    explicit FLFrame(QObject* parent = nullptr);
    FLFrame(const FLFrame &_other);
    FLFrame &operator=(const FLFrame &_other);
    ~FLFrame();

    qint64 index() const;
    void setIndex(qint64 _value);

    qint64 keyFrameIndex() const;
    void setKeyFrameIndex(qint64 _value);

    qint64 prevIndex() const;
    void setPrevIndex(qint64 _value);

    bool isKeyFrame() const;
    void setKeyFrame(bool _value = true);

    Eigen::Matrix4f pose() const;
    //void transform(const Eigen::Matrix4f& _value = Eigen::Matrix4f::Identity());
    void setPose(const Eigen::Matrix4f& _value);

    quint64 timestamp() const;
    void setTimestamp(quint64 _value);

    Eigen::Matrix3f rotation() const;
    Eigen::Vector3f translation() const;

    pcl::PointCloud<LineSegment>::Ptr lines() const;
	pcl::PointCloud<pcl::PointXYZINormal>::Ptr meanPointCloud() const;

private:
    void reproject(const Eigen::Matrix4f& pose);

private:
    QSharedDataPointer<FLFrameData> m_data;
};

#endif // FLFRAME_H