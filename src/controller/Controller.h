#ifndef CONTROLLER_H
#define CONTROLLER_H

#include <QObject>

#include "common/Frame.h"

class Controller : public QObject
{
    Q_OBJECT
public:
    explicit Controller(QObject *parent = nullptr);

    static Controller *current();

    static void destroy();

    virtual bool supportRandomAccessing() const = 0;

    virtual void fetchNext() = 0;

    virtual void moveTo(int frameIndex) = 0;

    virtual void skip(int frameNumbers) = 0;

    virtual void reset() = 0;

signals:
    void frameFetched(Frame& frame);

public slots:

private:
    static Controller *m_current;
};

#endif // CONTROLLER_H
