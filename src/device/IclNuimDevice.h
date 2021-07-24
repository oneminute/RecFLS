#ifndef ICLNUIMDEVICE_H
#define ICLNUIMDEVICE_H

#include "Device.h"

#include <QObject>

class IclNuimDevice : public Device
{
    Q_OBJECT
public:
    IclNuimDevice(QObject* parent = nullptr);

    // Device interface
public:
    virtual QString name() const;
    virtual bool open();
    virtual void close();
    virtual bool supportRandomAccessing();
    virtual void skip(int skipCount);
    virtual Frame getFrame(int frameIndex);
    virtual Frame fetchNext();
	virtual void start() override;
    virtual quint64 totalFrames() { return m_frameCount; }

private:
    quint64 m_frameCount;
    QList<int> m_frameIndices;
    QList<QString> m_depthFiles;
    QList<QString> m_rgbFiles;

    QString m_listFile;
    QString m_depthFolderName;
    QString m_rgbFolderName;

    int m_currentIndex;
};

#endif // ICLNUIMDEVICE_H
