#ifndef FRAME_H
#define FRAME_H

#include <QObject>
#include <QSharedDataPointer>

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

signals:

public slots:

private:
    QSharedDataPointer<FrameData> data;
};

#endif // FRAME_H
