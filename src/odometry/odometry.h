#ifndef ODOMETRY_H
#define ODOMETRY_H

#include <QObject>

#include "common/Frame.h"

class Odometry : public QObject
{
    Q_OBJECT
public:
    explicit Odometry(QObject *parent = nullptr);

    void process(Frame& frame);

signals:

protected:
    virtual bool beforeProcessing(Frame& frame) = 0;
    virtual void doProcessing(Frame& frame) = 0;
    virtual void afterProcessing(Frame& frame) = 0;

};

#endif // ODOMETRY_H
