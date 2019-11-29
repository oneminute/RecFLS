#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <QObject>

#include "common/Frame.h"

class Controller : public QObject
{
    Q_OBJECT
public:
    explicit Controller(QObject *parent = nullptr);

    virtual QString name() const = 0;

    virtual bool supportRandomAccessing() const = 0;

    virtual void fetchNext() = 0;

    virtual void moveTo(int frameIndex) = 0;

    virtual void skip(int frameNumbers) = 0;

    virtual void reset() = 0;

    virtual Frame getFrame(int frameIndex) = 0;

signals:
    void frameFetched(Frame& frame);

public slots:

private:
};

#endif // CONTROLLER_H
