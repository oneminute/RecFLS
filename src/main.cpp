#include "recfls.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    RecFLS w;
    w.show();

    return app.exec();
}

