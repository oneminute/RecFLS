#ifndef SENSORREADERDEVICE_H
#define SENSORREADERDEVICE_H

#include "Device.h"

#include <QObject>

class SensorReaderDevice : public Device
{
    Q_OBJECT
public:
    SensorReaderDevice();

    // Device interface
public:
    virtual QString name() const;
    virtual bool open();
    virtual void close();
    virtual bool supportRandomAccessing();
    virtual void skip(int skipCount);
    virtual Frame getFrame(int frameIndex);
    virtual void fetchNext();

private:
    quint64 m_frameCount;
    QList<Frame> m_frames;
    quint64 m_imuFrameCount;

    int m_currentIndex;
};

#endif // SENSORREADERDEVICE_H
