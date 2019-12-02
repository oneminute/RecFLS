#ifndef IMUFRAME_H
#define IMUFRAME_H

#include <QObject>
#include <QSharedDataPointer>

#include <Eigen/Core>
#include <Eigen/Dense>

class ImuFrameData;

class ImuFrame : public QObject
{
    Q_OBJECT
public:
    explicit ImuFrame(QObject *parent = nullptr);
    ImuFrame(const ImuFrame &);
    ImuFrame &operator=(const ImuFrame &);
    ~ImuFrame();

signals:

public slots:

private:
    QSharedDataPointer<ImuFrameData> data;
};

#endif // IMUFRAME_H
