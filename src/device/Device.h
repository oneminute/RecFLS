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

    virtual void fetchNext() = 0;

    virtual quint64 totalFrames() = 0;

    void initRectifyMap();

    cv::Mat undistortImage(const cv::Mat &in);

    cv::Mat alignDepthToColor(const cv::Mat &depthMat, const cv::Mat &colorMat);

    float fx() const;

    float fy() const;

    float cx() const;

    float cy() const;

    float depthShift() const;

    static Device* createDevice();

signals:
    void frameFetched(Frame& frame);
    void reachEnd();

public slots:

protected:
    Eigen::Matrix4f m_colorIntrinsic;
    Eigen::Matrix4f m_colorExtrinsic;
    Eigen::Matrix4f m_depthIntrinsic;
    Eigen::Matrix4f m_depthExtrinsic;

    Eigen::VectorXf m_distCoeffs;
    Eigen::Matrix3f m_R;

    QSize m_colorSize;
    QSize m_depthSize;
    float m_depthShift;

    cv::Mat m_rectifyMap1;
    cv::Mat m_rectifyMap2;
};

#endif // DEVICE_H
