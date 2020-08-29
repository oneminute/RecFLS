#include "RelInformation.h"

#include <QSharedData>

class RelInformationData : public QSharedData
{
public:
    RelInformationData()
    {}

    RelInformation::KeyPair key;
    Eigen::Quaternionf rot;
    Eigen::Vector3f trans;
    float error;
};

RelInformation::RelInformation(QObject* parent)
    : m_data(new RelInformationData)
{}

RelInformation::RelInformation(const RelInformation& other)
    : m_data(other.m_data)
{}

RelInformation::~RelInformation()
{
}

void RelInformation::setKey(qint64 from, qint64 to)
{
    m_data->key = KeyPair(from, to);
}

RelInformation::KeyPair RelInformation::key() const
{
    return m_data->key;
}

qint64 RelInformation::from() const
{
    return m_data->key.first;
}

qint64 RelInformation::to() const
{
    return m_data->key.second;
}

void RelInformation::setTransform(const Eigen::Matrix4f& transform)
{
    Eigen::Matrix3f rotM = transform.topLeftCorner(3, 3);
    m_data->trans = transform.topRightCorner(3, 1);
    m_data->rot = Eigen::Quaternionf(rotM);
}

Eigen::Matrix3f RelInformation::rotationMatrix() const
{
    return m_data->rot.toRotationMatrix();
}

Eigen::Quaternionf RelInformation::quaternion() const
{
    return m_data->rot;
}

Eigen::Vector3f RelInformation::translation() const
{
    return m_data->trans;
}

float RelInformation::error() const
{
    return m_data->error;
}

void RelInformation::setError(float error)
{
    m_data->error = error;
}

float RelInformation::confidence() const
{
    return 1.f / m_data->error;
}

Eigen::Matrix<float, 6, 6> RelInformation::information() const
{
    InformationMatrix inforMat(InformationMatrix::Identity());
    inforMat *= confidence();
    return inforMat;
}

RelInformation& RelInformation::operator=(const RelInformation& other)
{
    if (this != &other)
        m_data.operator=(other.m_data);
    return *this;
}