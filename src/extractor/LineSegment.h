#ifndef LINESEGMENT_H
#define LINESEGMENT_H

#include <QObject>
#include <QSharedDataPointer>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <pcl/common/common.h>
#include <pcl/features/eigen.h>
#include <pcl/search/kdtree.h>

#include <opencv2/opencv.hpp>

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

    cv::Point2f start2d() const;

    void setStart2d(const cv::Point2f& _value);

    cv::Point2f end2d() const;

    void setEnd2d(const cv::Point2f& _value);

    float length() const;

    Eigen::Vector3f direction() const;

    int index() const;

    void setIndex(int index);

    void generateShotDescriptor(float minLength, float maxLength, Eigen::Vector3f minPoint, Eigen::Vector3f maxPoint);

    void generateDescriptor(const Eigen::Matrix3f& rot = Eigen::Matrix3f::Identity(), const Eigen::Vector3f& trans = Eigen::Vector3f::Zero());

    int shortDescriptorSize() const;

    void calculateColorAvg(const cv::Mat& mat);

    void drawColorLine(cv::Mat& mat);

    int segmentNo() const;

    void setSegmentNo(int segmentNo);

    bool available() const;

    void reverse();

    bool similarDirection(const LineSegment &other, float &angle, float threshold);

    bool similarDirection(const LineSegment &other, float threshold);

    void applyAnotherLineDirection(const LineSegment& other);

    float angleToAnotherLine(const LineSegment &other);

    Eigen::Matrix<float, 1, 10> shortDescriptor() const;

    //Eigen::VectorXf longDescriptor() const;

    float averageDistance(const LineSegment &other);

    float pointDistance(const Eigen::Vector3f &point);

    Eigen::Vector3f closedPointOnLine(const Eigen::Vector3f &point);

    double red() const;

    double green() const;

    double blue() const;

signals:


private:
    QSharedDataPointer<LineSegmentData> data;
};

template<>
struct pcl::DefaultPointRepresentation<LineSegment>: public pcl::PointRepresentation<LineSegment>
{
public:
    DefaultPointRepresentation()
    {
        nr_dimensions_ = 10;
    }

    void copyToFloatArray(const LineSegment& l, float* out) const override
    {
        for (int i = 0; i < nr_dimensions_; i++)
        {
            out[i] = l.shortDescriptor()[i];
        }
    }

};

#endif // LINESEGMENT_H
