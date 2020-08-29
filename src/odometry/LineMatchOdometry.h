#ifndef LINEMATCHODOMETRY_H
#define LINEMATCHODOMETRY_H

#include "Odometry.h"

class LineMatchOdometry : public Odometry
{
    Q_OBJECT
public:
    explicit LineMatchOdometry(QObject* parent = nullptr);

    // Inherited via Odometry
    virtual void doProcessing(Frame& frame) override;
    virtual void afterProcessing(Frame& frame) override;
    virtual bool beforeProcessing(Frame& frame);

private:
};

#endif //LINEMATCHODOMETRY_H
