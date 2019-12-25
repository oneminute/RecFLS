#include "ui/MainWindow.h"
#include <QApplication>
#include <QDebug>
#include "common/Parameters.h"
#include "controller/Controller.h"
#include "controller/FrameStepController.h"
#include "device/Device.h"
#include "device/SensorReaderDevice.h"
#include "util/Utils.h"

#include <pcl/gpu/containers/initialization.h>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    pcl::gpu::setDevice(0);
    pcl::gpu::printShortCudaDeviceInfo(0);
//    pcl::gpu::printCudaDeviceInfo(0);

    Utils::registerTypes();

    Parameters::Global().load();

    qDebug() << "version:" << Parameters::Global().version();

    Device *device = new SensorReaderDevice;
    Controller *controller = new  FrameStepController(device);

    MainWindow w;
    w.setController(controller);
    w.show();

    int result = app.exec();

    delete device;

    return result;
}

