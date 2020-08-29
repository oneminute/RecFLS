#include "ImuFrame.h"

class ImuFrameData : public QSharedData
{
public:
    ImuFrameData()
        : frameIndex(-1)
    {}

    qint64 frameIndex;
    Eigen::Vector3d rotationRate;
    Eigen::Vector3d acceleration;
    Eigen::Vector3d magneticField;
    Eigen::Vector3d attitude;
    Eigen::Vector3d gravity;
    quint64 timeStamp;
};

ImuFrame::ImuFrame(QObject *parent) : QObject(parent), data(new ImuFrameData)
{

}

ImuFrame::ImuFrame(const ImuFrame &rhs) : data(rhs.data)
{

}

ImuFrame &ImuFrame::operator=(const ImuFrame &rhs)
{
    if (this != &rhs)
        data.operator=(rhs.data);
    return *this;
}

ImuFrame::~ImuFrame()
{

}
