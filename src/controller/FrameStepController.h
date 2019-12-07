#ifndef FRAMESTEPCONTROLLER_H
#define FRAMESTEPCONTROLLER_H

#include <QObject>
#include "Controller.h"

class FrameStepController : public Controller
{
    Q_OBJECT

public:
    FrameStepController(Device *device, QObject *parent = nullptr);

    // Controller interface
public:
    virtual QString name() const override;
    virtual bool open() override;
    virtual void close() override;
    virtual void fetchNext() override;
    virtual void moveTo(int frameIndex) override;
    virtual void skip(int frameNumbers) override;
    virtual void reset() override;
    virtual Frame getFrame(int frameIndex) override;

private slots:
    void onFrameFetched(Frame &frame);
};

#endif // FRAMESTEPCONTROLLER_H
