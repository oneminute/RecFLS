#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QScopedPointer>

#include "controller/Controller.h"
#include "CloudViewer.h"

class ToolWindowBoundaryExtractor;
class ToolWindowLineExtractor;
class ToolWindowLineMatcher;

class ToolWindowFusedLineExtractor;
class PreferencesWindow;

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
    
    void onActionToolWindowFusedLineExtractor();
    void onActionSaveCurrentFrame();
    void onActionPreferences();
    void onActionLineViewer();

    void onFrameFetched(Frame& frame);

private:
    QScopedPointer<Ui::MainWindow> m_ui;
    QScopedPointer<ToolWindowLineExtractor> m_toolWindowLineExtractor;
    QScopedPointer<ToolWindowLineMatcher> m_toolWindowLineMatcher;
    QScopedPointer<ToolWindowBoundaryExtractor> m_toolWindowBoundaryExtractor;
    
    QScopedPointer<ToolWindowFusedLineExtractor> m_fusedLineExtractor;
    QScopedPointer<PreferencesWindow> m_preferencesWindow;

    Controller *m_controller;
    CloudViewer *m_cloudViewer;
};

#endif // MAINWINDOW_H
