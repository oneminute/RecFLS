#include "ui/MainWindow.h"
#include <QApplication>
#include <QDebug>
#include "common/Parameters.h"
#include "controller/Controller.h"
#include "device/Device.h"
#include "device/SensorReaderDevice.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    Parameters::Global().load();

    qDebug() << "version:" << Parameters::Global().version();

    Device *device = new SensorReaderDevice;
    device->open();

    MainWindow w;
    w.show();

    int result = app.exec();

    delete device;

    return result;
}

