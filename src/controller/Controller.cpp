#include "Controller.h"
#include "common/Parameters.h"
#include "FrameStepController.h"

Controller *Controller::m_current(nullptr);

Controller::Controller(QObject *parent) : QObject(parent)
{

}

Controller *Controller::current()
{
    if (m_current)
    {

    }
    else
    {
        QString currentClassName = Parameters::Global().stringValue("current_controller", "FrameStepController", "Controller");

        if (currentClassName == "FrameStepController")
        {
            m_current = new FrameStepController;
        }
    }
    return m_current;
}

void Controller::destroy()
{
    if (m_current)
    {
        delete m_current;
        m_current = nullptr;
    }
}
