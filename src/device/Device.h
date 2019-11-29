#ifndef DEVICE_H
#define DEVICE_H

#include <QObject>
#include <QSize>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/Dense>

#include "common/Frame.h"

class Device : public QObject
{
    Q_OBJECT
public:
    explicit Device(QObject *parent = nullptr);

    virtual QString name() const = 0;

    virtual bool open() = 0;

    virtual void close() = 0;

    virtual bool supportRandomAccessing() = 0;

    virtual void skip(int skipCount) = 0;

    virtual Frame getFrame(int frameIndex) = 0;

signals:
    void frameFetched(const Frame& frame);

public slots:

protected:
    Eigen::Matrix4f m_colorIntrinsic;
    Eigen::Matrix4f m_colorExtrinsic;
    Eigen::Matrix4f m_depthIntrinsic;
    Eigen::Matrix4f m_depthExtrinsic;

    Frame::COMPRESSION_TYPE_COLOR m_colorCompressionType;
    Frame::COMPRESSION_TYPE_DEPTH m_depthCompressionType;

    QSize m_colorSize;
    QSize m_depthSize;
    float m_depthShift;
};

#endif // DEVICE_H
