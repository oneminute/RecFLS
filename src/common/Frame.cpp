#include "Frame.h"
#include "util/Utils.h"

class FrameData : public QSharedData
{
public:
    FrameData()
        : deviceFrameIndex(-1)
        , frameIndex(-1)
        , keyFrameIndex(-1)
        , cameraToWorld(Eigen::Matrix4f::Zero())
        , timeStampColor(0)
        , timeStampDepth(0)
        , colorCompressed()
        , depthCompressed()
    {}

    qint64 deviceFrameIndex;
    qint64 frameIndex;
    qint64 keyFrameIndex;
    Eigen::Matrix4f cameraToWorld;
    quint64 timeStampColor;
    quint64 timeStampDepth;
    QByteArray colorCompressed;
    QByteArray depthCompressed;
    cv::Mat colorMat;
    cv::Mat depthMat;
};

Frame::Frame(QObject *parent) : QObject(parent), data(new FrameData)
{

}

Frame::Frame(const Frame &rhs) : data(rhs.data)
{

}

Frame &Frame::operator=(const Frame &rhs)
{
    if (this != &rhs)
        data.operator=(rhs.data);
    return *this;
}

Frame::~Frame()
{

}

void Frame::setDeviceFrameIndex(qint64 index)
{
    data->deviceFrameIndex = index;
}

qint64 Frame::deviceFrameIndex() const
{
    return data->deviceFrameIndex;
}

void Frame::setFrameIndex(qint64 index)
{
    data->frameIndex = index;
}

qint64 Frame::frameIndex() const
{
    return data->frameIndex;
}

void Frame::setKeyFrameIndex(qint64 index)
{
    data->keyFrameIndex = index;
}

qint64 Frame::keyFrameIndex() const
{
    return data->keyFrameIndex;
}

void Frame::setCameraToWorld(Eigen::Matrix4f cameraToWorld)
{
    data->cameraToWorld = cameraToWorld;
}

Eigen::Matrix4f Frame::cameraToWorld() const
{
    return data->cameraToWorld;
}

void Frame::setTimeStampColor(quint64 timeStampColor)
{
    data->timeStampColor = timeStampColor;
}

quint64 Frame::timeStampColor() const
{
    return data->timeStampColor;
}

void Frame::setTimeStampDepth(quint64 timeStampDepth)
{
    data->timeStampDepth = timeStampDepth;
}

quint64 Frame::timeStampDepth() const
{
    return data->timeStampDepth;
}

void Frame::setColorCompressed(const QByteArray &colorCompressed)
{
    data->colorCompressed = colorCompressed;
}

QByteArray Frame::colorCompressed() const
{
    return data->colorCompressed;
}

void Frame::setDepthCompressed(const QByteArray &depthCompressed)
{
    data->depthCompressed = depthCompressed;
}

QByteArray Frame::depthCompressed() const
{
    return data->depthCompressed;
}

void Frame::setColorMat(const cv::Mat &colorMat)
{
    data->colorMat = colorMat;
}

cv::Mat Frame::colorMat() const
{
    return data->colorMat;
}

void Frame::setDepthMat(const cv::Mat &depthMat)
{
    data->depthMat = depthMat;
}

cv::Mat Frame::depthMat() const
{
    return data->depthMat;
}

QDataStream &Frame::load(QDataStream &in)
{
    for (int i = 0; i < 16; i++)
    {
        float v = 0;
        in >> v;
        data->cameraToWorld.data()[i] = v;
    }

    in >> data->timeStampColor;
    in >> data->timeStampDepth;

    quint64 colorBytes = 0;
    quint64 depthBytes = 0;
    in >> colorBytes >> depthBytes;

    data->colorCompressed.resize(static_cast<int>(colorBytes));
    data->depthCompressed.resize(static_cast<int>(depthBytes));
    in.readRawData(data->colorCompressed.data(), static_cast<int>(colorBytes));
    in.readRawData(data->depthCompressed.data(), static_cast<int>(depthBytes));
    return in;
}

void Frame::showInfo() const
{
    qDebug().noquote() << "device frame index:" << data->deviceFrameIndex << endl
                       << "       frame index:" << data->frameIndex << endl
                       << "   key frame index:" << data->keyFrameIndex << endl
                       << "   camera to world:" << endl << data->cameraToWorld
                       << "  color time stamp:" << data->timeStampColor << endl
                       << "  depth time stamp:" << data->timeStampDepth << endl
                       << " color frame bytes:" << data->colorCompressed.size() << endl
                       << " depth frame bytes:" << data->depthCompressed.size() << endl;
}

bool Frame::hasCompressedData() const
{
    return data->colorCompressed.size() != 0 || data->depthCompressed.size() != 0;
}

bool Frame::hasUncompressedData() const
{
    return !data->colorMat.empty() || !data->depthMat.empty();
}

void Frame::clearCompressedData()
{
    data->colorCompressed.clear();
    data->depthCompressed.clear();
}

void Frame::clearUncompressedData()
{
    data->colorMat = cv::Mat();
    data->depthMat = cv::Mat();
}

QDataStream &operator>>(QDataStream &in, Frame &frame)
{
    return frame.load(in);
}
