#ifndef FRAMESTEPCONTROLLER_H
#define FRAMESTEPCONTROLLER_H

#include <QObject>
#include "Controller.h"

class FrameStepController : public Controller
{
    Q_OBJECT

public:
    FrameStepController();

    // Controller interface
public:
    virtual QString name() const override;
    virtual bool supportRandomAccessing() const;
    virtual void fetchNext();
    virtual void moveTo(int frameIndex);
    virtual void skip(int frameNumbers);
    virtual void reset();
    virtual Frame getFrame(int frameIndex);
};

#endif // FRAMESTEPCONTROLLER_H
