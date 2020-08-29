#include "Controller.h"
#include "common/Parameters.h"
#include "DefaultController.h"

Controller::Controller(Device *device, QObject *parent)
    : QObject(parent)
    , m_device(device)
{
    Q_ASSERT(m_device);

//    connect(m_device, &Device::frameFetched, this, &Controller::frameFetched);
}

bool Controller::supportRandomAccessing() const
{
    return m_device->supportRandomAccessing();
}

