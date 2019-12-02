#ifndef FRAME_H
#define FRAME_H

#include <QObject>
#include <QSharedDataPointer>
#include <QDataStream>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

class FrameData;

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
    cv::Mat colorMat() const;

    void setDepthMat(const cv::Mat &depthMat);
    cv::Mat depthMat() const;

    QDataStream& load(QDataStream &in);

    void showInfo() const;

    bool hasCompressedData() const;

    bool hasUncompressedData() const;

    void clearCompressedData();

    void clearUncompressedData();

signals:

public slots:

private:
    QSharedDataPointer<FrameData> data;
};

QDataStream &operator>>(QDataStream &in, Frame &frame);

#endif // FRAME_H
