#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QScopedPointer>

#include "controller/Controller.h"
#include "CloudViewer.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

    void setController(Controller *controller);
    Controller *controller() const;

    void setAspectRatio(int w, int h);

private:
    void onActionNextFrame();
    void onActionPreviousFrame(bool);
    void onActionOpenDevice(bool checked);
    void onActionPause();
    void onActionCloseDevice();

    void onFrameFetched(Frame& frame);

private:
    QScopedPointer<Ui::MainWindow> m_ui;

    Controller *m_controller;
    CloudViewer *m_cloudViewer;
};

#endif // MAINWINDOW_H
