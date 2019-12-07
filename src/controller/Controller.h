#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <QObject>

#include "common/Frame.h"
#include "device/Device.h"

class Controller : public QObject
{
    Q_OBJECT
public:
    explicit Controller(Device *device, QObject *parent = nullptr);

    virtual QString name() const = 0;

    bool supportRandomAccessing() const;

    virtual bool open() = 0;

    virtual void close() = 0;

    virtual void fetchNext() = 0;

    virtual void moveTo(int frameIndex) = 0;

    virtual void skip(int frameNumbers) = 0;

    virtual void reset() = 0;

    virtual Frame getFrame(int frameIndex) = 0;

signals:
    void frameFetched(Frame& frame);

public slots:

protected:
    Device *m_device;
    QList<QPair<QString, cv::Mat>> m_filteredMat;
};

#endif // CONTROLLER_H
