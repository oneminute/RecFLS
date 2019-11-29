#ifndef FRAME_H
#define FRAME_H

#include <QObject>
#include <QSharedDataPointer>

class FrameData;

class Frame : public QObject
{
    Q_OBJECT
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
