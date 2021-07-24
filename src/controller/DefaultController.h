#ifndef FRAMESTEPCONTROLLER_H
#define DEFAULTCONTROLLER_H

#include <QObject>
#include "Controller.h"

class DefaultController : public Controller
{
    Q_OBJECT

public:
    DefaultController(Device *device, QObject *parent = nullptr);

    // Controller interface
public:
    virtual QString name() const override;
    virtual bool open() override;
    virtual void close() override;
    virtual void fetchNext() override;
	virtual void start() override;
    virtual void moveTo(int frameIndex) override;
    virtual void skip(int frameNumbers) override;
    virtual void reset() override;
    virtual Frame getFrame(int frameIndex) override;

    virtual void saveCurrentFrame() override;

private slots:
    void onFrameFetched(Frame &frame);

private:

};

#endif // FRAMESTEPCONTROLLER_H
