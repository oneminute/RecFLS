#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QScopedPointer>

#include "controller/Controller.h"
#include "CloudViewer.h"
#include "ToolWindowLineExtractor.h"

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
    void onActionToolWindowLineExtractor();
    void onActionSaveCurrentFrame();

    void onFrameFetched(Frame& frame);

private:
    QScopedPointer<Ui::MainWindow> m_ui;
    QScopedPointer<ToolWindowLineExtractor> m_toolWindowLineExtractor;

    Controller *m_controller;
    CloudViewer *m_cloudViewer;
};

#endif // MAINWINDOW_H
