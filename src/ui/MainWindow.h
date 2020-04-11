#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QScopedPointer>

#include "controller/Controller.h"
#include "CloudViewer.h"
#include "ToolWindowLineExtractor.h"
#include "ToolWindowLineMatcher.h"
#include "ToolWindowBoundaryExtractor.h"
#include "PreferencesWindow.h"

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
    void onActionToolWindowLineMatcher();
    void onActionToolWindowBoundaryExtractor();
    void onActionSaveCurrentFrame();
    void onActionPreferences();

    void onFrameFetched(Frame& frame);

private:
    QScopedPointer<Ui::MainWindow> m_ui;
    QScopedPointer<ToolWindowLineExtractor> m_toolWindowLineExtractor;
    QScopedPointer<ToolWindowLineMatcher> m_toolWindowLineMatcher;
    QScopedPointer<ToolWindowBoundaryExtractor> m_toolWindowBoundaryExtractor;
    QScopedPointer<PreferencesWindow> m_preferencesWindow;

    Controller *m_controller;
    CloudViewer *m_cloudViewer;
};

#endif // MAINWINDOW_H
