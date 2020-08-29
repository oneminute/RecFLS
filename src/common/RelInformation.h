#ifndef RELINFORMATION_H
#define RELINFORMATION_H

#include <QObject>
#include <QSharedDataPointer>
#include <QPair>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <extractor/LineSegment.h>
#include <pcl/common/common.h>

class RelInformationData;

class RelInformation : public QObject
{
    Q_OBJECT
public:
    typedef QPair<qint64, qint64> KeyPair;
    typedef Eigen::Matrix<float, 6, 6> InformationMatrix;

    explicit RelInformation(QObject* parent = nullptr);
    RelInformation(const RelInformation& other);
    RelInformation& operator=(const RelInformation& other);

    ~RelInformation();

    void setKey(qint64 from, qint64 to);

    KeyPair key() const;

    qint64 from() const;

    qint64 to() const;

    void setTransform(const Eigen::Matrix4f& transform);

    Eigen::Matrix3f rotationMatrix() const;

    Eigen::Quaternionf quaternion() const;

    Eigen::Vector3f translation() const;

    float error() const;

    void setError(float error);

    float confidence() const;

    InformationMatrix information() const;

private:
    

    QSharedDataPointer<RelInformationData> m_data;
};

#endif // RELINFORMATION_H
