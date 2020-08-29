#ifndef FRAME_H
#define FRAME_H

#include <QObject>
#include <QSharedDataPointer>
#include <QDataStream>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <pcl/common/common.h>

class FrameData;
class Device;

class Frame : public QObject
{
    Q_OBJECT
public:
    enum COMPRESSION_TYPE_COLOR {
        TYPE_COLOR_UNKNOWN = -1,
        TYPE_RAW = 0,
        TYPE_PNG = 1,
        TYPE_JPEG = 2
    };

    enum COMPRESSION_TYPE_DEPTH {
        TYPE_DEPTH_UNKNOWN = -1,
        TYPE_RAW_USHORT = 0,
        TYPE_ZLIB_USHORT = 1,
        TYPE_OCCI_USHORT = 2
    };

public:
    explicit Frame(QObject *parent = nullptr);
    Frame(const Frame &);
    Frame &operator=(const Frame &);
    ~Frame();

    void setDeviceFrameIndex(qint64 index);
    qint64 deviceFrameIndex() const;

    void setFrameIndex(qint64 index);
    qint64 frameIndex() const;

    void setKeyFrameIndex(qint64 index);
    qint64 keyFrameIndex() const;

    void setCameraToWorld(Eigen::Matrix4f cameraToWorld);
    Eigen::Matrix4f cameraToWorld() const;

    void setTimeStampColor(quint64 timeStampColor);
    quint64 timeStampColor() const;

    void setTimeStampDepth(quint64 timeStampDepth);
    quint64 timeStampDepth() const;

    void setColorCompressed(const QByteArray &colorCompressed);
    QByteArray colorCompressed() const;

    void setDepthCompressed(const QByteArray &depthCompressed);
    QByteArray depthCompressed() const;

    void setColorMat(const cv::Mat &colorMat);
    cv::Mat colorMat();

    void setDepthMat(const cv::Mat &depthMat);
    cv::Mat depthMat();

    Frame::COMPRESSION_TYPE_COLOR getColorCompressionType() const;
    void setColorCompressionType(const Frame::COMPRESSION_TYPE_COLOR &value);

    Frame::COMPRESSION_TYPE_DEPTH getDepthCompressionType() const;
    void setDepthCompressionType(const Frame::COMPRESSION_TYPE_DEPTH &value);

    int getColorWidth() const;
    void setColorWidth(int value);

    int getColorHeight() const;
    void setColorHeight(int value);

    int getDepthWidth() const;
    void setDepthWidth(int value);

    int getDepthHeight() const;
    void setDepthHeight(int value);

    Device* getDevice() const;
    void setDevice(Device* device);

    QDataStream& load(QDataStream &in);

    void showInfo() const;

    bool hasCompressedData() const;

    bool hasUncompressedData() const;

    void clearCompressedData();

    void clearUncompressedData();

    bool isAvailable() const;

    cv::Mat undistortRGBImage();

    cv::Mat alignDepthToColor();

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr getCloud(std::vector<int>& indices, float minDepth = 0.4f, float maxDepth = 8.0f);

signals:

public slots:

private:
    QSharedDataPointer<FrameData> data;
};

QDataStream &operator>>(QDataStream &in, Frame &frame);

#endif // FRAME_H
