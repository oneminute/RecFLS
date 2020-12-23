#include "FLFrame.h"

class FLFrameData : public QSharedData
{
public:
    FLFrameData(
        qint64 _index = -1
        , qint64 _keyFrameIndex = -1
        , qint64 _preIndex = -1
        , bool _isKeyFrame = false
        , Eigen::Matrix4f _pose = Eigen::Matrix4f::Identity()
        , pcl::PointCloud<LineSegment>::Ptr _lines = pcl::PointCloud<LineSegment>::Ptr(new pcl::PointCloud<LineSegment>))
        : index(_index)
        , keyFrameIndex(_keyFrameIndex)
        , prevIndex(_preIndex)
        , isKeyFrame(_isKeyFrame)
        , pose(_pose)
        , lines(_lines)
    { }

    qint64 index;
    qint64 keyFrameIndex;
    qint64 prevIndex;
    bool isKeyFrame;
    Eigen::Matrix4f pose;
    quint64 timestamp;
    pcl::PointCloud<LineSegment>::Ptr lines;

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

FLFrame::FLFrame(QObject* parent)
    : QObject(parent)
    , m_data(new FLFrameData)
{

}

FLFrame::FLFrame(const FLFrame& _other) : m_data(_other.m_data)
{

}

FLFrame& FLFrame::operator=(const FLFrame& _other)
{
    if (this != &_other)
        m_data.operator=(_other.m_data);
    return *this;
}

FLFrame::~FLFrame()
{

}

qint64 FLFrame::index() const
{
    return m_data->index;
}

void FLFrame::setIndex(qint64 _value)
{
    m_data->index = _value;
}

qint64 FLFrame::keyFrameIndex() const
{
    return m_data->keyFrameIndex;
}

void FLFrame::setKeyFrameIndex(qint64 _value)
{
    m_data->keyFrameIndex = _value;
}

qint64 FLFrame::prevIndex() const
{
    return m_data->prevIndex;
}

void FLFrame::setPrevIndex(qint64 _value)
{
    m_data->prevIndex = _value;
}

bool FLFrame::isKeyFrame() const
{
    return m_data->isKeyFrame;
}

void FLFrame::setKeyFrame(bool _value)
{
    m_data->isKeyFrame = _value;
}

Eigen::Matrix4f FLFrame::pose() const
{
    return m_data->pose;
}

//void FLFrame::transform(const Eigen::Matrix4f& _value)
//{
//    m_data->pose = _value * m_data->pose;
//    reproject(_value);
//}

void FLFrame::setPose(const Eigen::Matrix4f& _value)
{
    m_data->pose = _value * m_data->pose;
    reproject(_value);
}

quint64 FLFrame::timestamp() const
{
    return m_data->timestamp;
}

void FLFrame::setTimestamp(quint64 _value)
{
    m_data->timestamp = _value;
}

Eigen::Matrix3f FLFrame::rotation() const
{
    return m_data->pose.topLeftCorner(3, 3);
}

Eigen::Vector3f FLFrame::translation() const
{
    return m_data->pose.topRightCorner(3, 1);
}

pcl::PointCloud<LineSegment>::Ptr FLFrame::lines() const
{
    return m_data->lines;
}

void FLFrame::reproject(const Eigen::Matrix4f& pose)
{
    Eigen::Matrix3f rot = pose.topLeftCorner(3, 3);
    Eigen::Vector3f trans = pose.topRightCorner(3, 1);

    for (pcl::PointCloud<LineSegment>::iterator i = m_data->lines->begin(); i != m_data->lines->end(); i++)
    {
        i->reproject(rot, trans);
    }
}
