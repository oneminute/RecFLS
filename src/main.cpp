#include "ui/MainWindow.h"
#include <QApplication>
#include <QDebug>
#include "common/Parameters.h"
#include "controller/Controller.h"

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);

    Parameters::Global().load();

    qDebug() << "version:" << Parameters::Global().stringValue("version", "0.1.0");

    Controller::current();

    MainWindow w;
    w.show();

    int result = app.exec();

    Controller::destroy();
    return result;
}

