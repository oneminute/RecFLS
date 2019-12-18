#ifndef LINESEGMENT_H
#define LINESEGMENT_H

#include <QObject>
#include <QSharedDataPointer>

#include <Eigen/Core>
#include <Eigen/Dense>

class LineSegmentData;

class LineSegment : public QObject
{
    Q_OBJECT
public:
    explicit LineSegment(const Eigen::Vector3f &start = Eigen::Vector3f(0, 0, 0),
                         const Eigen::Vector3f &end = Eigen::Vector3f(0, 0, 0),
                         int segmentNo = -1, QObject *parent = nullptr);
    LineSegment(const LineSegment &);
    LineSegment &operator=(const LineSegment &);
    ~LineSegment();

    Eigen::Vector3f start() const;

    void setStart(const Eigen::Vector3f &pt);

    Eigen::Vector3f end() const;

    void setEnd(const Eigen::Vector3f &pt);

    Eigen::Vector3f middle() const;

    float length() const;

    Eigen::Vector3f direction() const;

    void generateShotDescriptor(float minLength, float maxLength, Eigen::Vector3f minPoint, Eigen::Vector3f maxPoint);

    int segmentNo() const;

    void setSegmentNo(int segmentNo);

    bool available() const;

    void reverse();

    Eigen::VectorXf shortDescriptor() const;

    Eigen::VectorXf longDescriptor() const;

signals:


private:
    QSharedDataPointer<LineSegmentData> data;
};

#endif // LINESEGMENT_H
