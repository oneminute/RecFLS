#include "Frame.h"
#include "util/Utils.h"

#include "stb_image.h"
#include "device/Device.h"

class FrameData : public QSharedData
{
public:
    FrameData()
        : deviceFrameIndex(-1)
        , frameIndex(-1)
        , keyFrameIndex(-1)
        , colorWidth(0)
        , colorHeight(0)
        , cameraToWorld(Eigen::Matrix4f::Zero())
        , timeStampColor(0)
        , timeStampDepth(0)
        , colorCompressed()
        , depthCompressed()
        , colorCompressionType(Frame::TYPE_COLOR_UNKNOWN)
        , depthCompressionType(Frame::TYPE_DEPTH_UNKNOWN)
        , device(nullptr)
    {}

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    qint64 deviceFrameIndex;
    qint64 frameIndex;
    qint64 keyFrameIndex;
    int colorWidth;
    int colorHeight;
    int depthWidth;
    int depthHeight;
    Eigen::Matrix4f cameraToWorld;
    quint64 timeStampColor;
    quint64 timeStampDepth;
    QByteArray colorCompressed;
    QByteArray depthCompressed;
    cv::Mat colorMat;
    cv::Mat depthMat;
    Frame::COMPRESSION_TYPE_COLOR colorCompressionType;
    Frame::COMPRESSION_TYPE_DEPTH depthCompressionType;
    QList<QPair<QString, qreal>> durations;
    Device* device;
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

cv::Mat Frame::colorMat()
{
    if (data->colorMat.empty())
    {
        if (data->colorCompressed.size() > 0)
        {
            if (data->colorCompressionType == TYPE_RAW)
            {
                data->colorMat = cv::Mat(data->colorHeight, data->colorWidth, CV_8UC3, const_cast<char*>(data->colorCompressed.data()));
            }
            else
            {
                cv::Mat compressed(1, data->colorCompressed.size(), CV_8U, const_cast<char*>(data->colorCompressed.data()));
                data->colorMat = cv::imdecode(compressed, cv::IMREAD_UNCHANGED);
            }
        }
    }
    return data->colorMat;
}

void Frame::setDepthMat(const cv::Mat &depthMat)
{
    data->depthMat = depthMat;
}

cv::Mat Frame::depthMat()
{
    if (data->depthMat.empty())
    {
        if (data->depthCompressed.size() > 0)
        {
            if (data->depthCompressionType == TYPE_RAW_USHORT)
            {
                data->depthMat = cv::Mat(data->depthHeight, data->depthWidth, CV_16U, const_cast<char*>(data->depthCompressed.data()));
            }
            else if (data->depthCompressionType == TYPE_ZLIB_USHORT)
            {
                char* res;
                int len;
                res = stbi_zlib_decode_malloc(data->depthCompressed.data(), data->depthCompressed.size(), &len);
                data->depthMat = cv::Mat(data->depthHeight, data->depthWidth, CV_16U);
                memcpy(data->depthMat.data, res, len);
                stbi_image_free(res);
            }
        }
    }
    return data->depthMat;
}

Frame::COMPRESSION_TYPE_COLOR Frame::getColorCompressionType() const
{
    return data->colorCompressionType;
}

void Frame::setColorCompressionType(const Frame::COMPRESSION_TYPE_COLOR &value)
{
    data->colorCompressionType = value;
}

Frame::COMPRESSION_TYPE_DEPTH Frame::getDepthCompressionType() const
{
    return data->depthCompressionType;
}

void Frame::setDepthCompressionType(const Frame::COMPRESSION_TYPE_DEPTH &value)
{
    data->depthCompressionType = value;
}

int Frame::getColorHeight() const
{
    return data->colorHeight;
}

void Frame::setColorHeight(int value)
{
    data->colorHeight = value;
}

int Frame::getColorWidth() const
{
    return data->colorWidth;
}

void Frame::setColorWidth(int value)
{
    data->colorWidth = value;
}

int Frame::getDepthHeight() const
{
    return data->depthHeight;
}

void Frame::setDepthHeight(int value)
{
    data->depthHeight = value;
}

Device* Frame::getDevice() const
{
    return data->device;
}

void Frame::setDevice(Device* device)
{
    data->device = device;
}

int Frame::getDepthWidth() const
{
    return data->depthWidth;
}

void Frame::setDepthWidth(int value)
{
    data->depthWidth = value;
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

bool Frame::isAvailable() const
{
    return data->frameIndex >= 0;
}

cv::Mat Frame::undistortRGBImage()
{
    return data->device->undistortImage(data->colorMat);
}

cv::Mat Frame::alignDepthToColor()
{
    return data->device->alignDepthToColor(data->depthMat, data->colorMat);
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr Frame::getCloud(pcl::Indices& indices, float minDepth, float maxDepth)
{
    indices.clear();
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);

    cv::Mat depth = depthMat();
    cv::Mat color = colorMat();
    depth.convertTo(depth, CV_32F);
    int count = 0;
    for(int i = 0; i < getDepthHeight(); i++) {
        for(int j = 0; j < getDepthWidth(); j++) {
            int index = i * getDepthWidth() + j;
            pcl::PointXYZRGB pt;
            float zValue = depth.at<float>(i, j);

            float x = 0, y = 0, z = 0;
            z = zValue / getDevice()->depthShift();
            x = (j - getDevice()->cx()) * z / getDevice()->fx();
            y = (i - getDevice()->cy()) * z / getDevice()->fy();

            pt.x = x;
            pt.y = y;
            pt.z = z;
            pt.b = color.at<cv::Vec3b>(cv::Point(j, i))[0];
            pt.g = color.at<cv::Vec3b>(cv::Point(j, i))[1];
            pt.r = color.at<cv::Vec3b>(cv::Point(j, i))[2];

            if (pt.z > minDepth && pt.z <= maxDepth) {
                indices.push_back(count++);
                cloud->push_back(pt);
            }
        }
    }
    cloud->is_dense = true;
    //cloud->width = getDepthWidth();
    //cloud->height =  getDepthHeight();
    cloud->width = cloud->points.size();
    cloud->height =  1;

    return cloud;
}

QDataStream &operator>>(QDataStream &in, Frame &frame)
{
    return frame.load(in);
}
