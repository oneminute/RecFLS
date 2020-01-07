#include "LineSegment.h"

#include <QtMath>

class LineSegmentData : public QSharedData
{
public:
    LineSegmentData(Eigen::Vector3f _start = Eigen::Vector3f(0, 0, 0), Eigen::Vector3f _end = Eigen::Vector3f(0, 0, 0), int _segmentNo = -1)
        : start(_start)
        , end(_end)
        , segmentNo(_segmentNo)
    {}

    Eigen::Vector3f start;
    Eigen::Vector3f end;
    int segmentNo;

    Eigen::VectorXf shotDescriptor;
    Eigen::VectorXf longDescriptor;
};

LineSegment::LineSegment(const Eigen::Vector3f &start, const Eigen::Vector3f &end, int segmentNo, QObject *parent)
    : QObject(parent)
    , data(new LineSegmentData(start, end, segmentNo))
{

}

LineSegment::LineSegment(const LineSegment &rhs) : data(rhs.data)
{

}

LineSegment &LineSegment::operator=(const LineSegment &rhs)
{
    if (this != &rhs)
        data.operator=(rhs.data);
    return *this;
}

LineSegment::~LineSegment()
{

}

Eigen::Vector3f LineSegment::start() const
{
    return data->start;
}

void LineSegment::setStart(const Eigen::Vector3f &pt)
{
    data->start = pt;
}

Eigen::Vector3f LineSegment::end() const
{
    return data->end;
}

void LineSegment::setEnd(const Eigen::Vector3f &pt)
{
    data->end = pt;
}

Eigen::Vector3f LineSegment::middle() const
{
    return (start() + end()) / 2;
}

float LineSegment::length() const
{
    return (end() - start()).norm();
}

Eigen::Vector3f LineSegment::direction() const
{
    return end() - start();
}

void LineSegment::generateShotDescriptor(float minLength, float maxLength, Eigen::Vector3f minPoint, Eigen::Vector3f maxPoint)
{
    Eigen::Vector3f s = start() - minPoint;
    Eigen::Vector3f m = middle() - minPoint;
    Eigen::Vector3f e = end() - minPoint;

    Eigen::Vector3f delta = maxPoint - minPoint;

    data->shotDescriptor.resize(1, 13);
    data->shotDescriptor[0] = s.x() / delta.x();
    data->shotDescriptor[1] = s.y() / delta.y();
    data->shotDescriptor[2] = s.z() / delta.z();
    data->shotDescriptor[3] = m.x() / delta.x();
    data->shotDescriptor[4] = m.y() / delta.y();
    data->shotDescriptor[5] = m.z() / delta.z();
    data->shotDescriptor[6] = e.x() / delta.x();
    data->shotDescriptor[7] = e.y() / delta.y();
    data->shotDescriptor[8] = e.z() / delta.z();

    Eigen::Vector3f dir = direction().normalized();
    data->shotDescriptor[9] = dir[0];
    data->shotDescriptor[10] = dir[1];
    data->shotDescriptor[11] = dir[2];
    data->shotDescriptor[12] = (length() - minLength) / (maxLength - minLength);
    data->shotDescriptor.normalize();
}

int LineSegment::segmentNo() const
{
    return data->segmentNo;
}

void LineSegment::setSegmentNo(int segmentNo)
{
    data->segmentNo = segmentNo;
}

bool LineSegment::available() const
{
    return length() > 0;
}

void LineSegment::reverse()
{
    Eigen::Vector3f tmp = data->start;
    data->start = data->end;
    data->end = tmp;
}

bool LineSegment::similarDirection(const LineSegment &other, float &angle, float threshold)
{
    angle = angleToAnotherLine(other);
    if (angle < threshold)
    {
        return true;
    }
    else if (angle > (M_PI - threshold))
    {
        angle = M_PI - angle;
        return true;
    }

    return false;
}

bool LineSegment::similarDirection(const LineSegment &other, float threshold)
{
    float angle = angleToAnotherLine(other);
    if (angle < threshold || angle > (M_PI - threshold))
        return true;
    return false;
}

void LineSegment::applyAnotherLineDirection(const LineSegment& other)
{
    if (direction().dot(other.direction()) < 0)
    {
        reverse();
    }
}

float LineSegment::angleToAnotherLine(const LineSegment &other)
{
    return qAcos(direction().normalized().dot(other.direction().normalized()));
}

Eigen::VectorXf LineSegment::shortDescriptor() const
{
    return data->shotDescriptor;
}

Eigen::VectorXf LineSegment::longDescriptor() const
{
    return data->longDescriptor;
}

float LineSegment::averageDistance(const LineSegment &other)
{
    float distM = pointDistance(other.middle());
    return distM;
}

float LineSegment::pointDistance(const Eigen::Vector3f &point)
{
    Eigen::Vector3f line = point - data->start;
    Eigen::Vector3f dir = direction().normalized();
    Eigen::Vector3f pointProj = data->start + dir * (line.dot(dir));
    float distance = qAbs((point - pointProj).norm());
    return distance;
}

Eigen::Vector3f LineSegment::closedPointOnLine(const Eigen::Vector3f &point)
{
    Eigen::Vector3f dir = direction().normalized();
    Eigen::Vector3f ev = point - middle();
    Eigen::Vector3f closedPoint = middle() + dir * (ev.dot(dir));
    return closedPoint;
}
