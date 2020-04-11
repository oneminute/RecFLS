#include <QApplication>
#include <QDebug>
#include "ui/MainWindow.h"
#include "common/Parameters.h"
#include "controller/Controller.h"
#include "controller/DefaultController.h"
#include "device/Device.h"
#include "device/SensorReaderDevice.h"
#include "util/Utils.h"
#include "util/StopWatch.h"

#include <pcl/gpu/containers/initialization.h>

#ifdef _MSC_VER
//#    ifdef NDEBUG
//#        pragma comment(linker, "/SUBSYSTEM:WINDOWS /ENTRY:mainCRTStartup")
//#    else
//#        pragma comment(linker, "/SUBSYSTEM:CONSOLE")
//#    endif
#    pragma comment(linker, "/SUBSYSTEM:CONSOLE")
#endif

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    pcl::gpu::setDevice(0);
    pcl::gpu::printShortCudaDeviceInfo(0);
//    pcl::gpu::printCudaDeviceInfo(0);

    Utils::registerTypes();

    //Parameters::Global().load();

    //Parameters::Global().setVersion("0.1.01");
    //qDebug() << "version:" << Parameters::Global().version();

    size_t size;
    cudaDeviceGetLimit(&size, cudaLimitMallocHeapSize);
    qDebug() << "cuda limit malloc heap size:" << size << "bytes";

    StopWatch::instance().start();

    Settings::load();
    Settings::save();

    Device *device = new SensorReaderDevice;
    Controller *controller = new  DefaultController(device);

    MainWindow* w = new MainWindow;
    w->setController(controller);
    w->show();

    int result = app.exec();

    StopWatch::instance().debugPrint();
    //Parameters::Global().save();
    delete device;
    delete w;

    return result;
}

