#ifndef TOOLWINDOWBOUNDARYEXTRACTOR_H
#define TOOLWINDOWBOUNDARYEXTRACTOR_H

#include <QMainWindow>
#include <QScopedPointer>

#include "ui/CloudViewer.h"
#include "ui/ImageViewer.h"
#include "extractor/BoundaryExtractor.h"
#include "extractor/DDBPLineExtractor.h"
#include "device/SensorReaderDevice.h"

namespace Ui {
class ToolWindowBoundaryExtractor;
}

class ToolWindowBoundaryExtractor : public QMainWindow
{
    Q_OBJECT

public:
    explicit ToolWindowBoundaryExtractor(QWidget *parent = nullptr);
    ~ToolWindowBoundaryExtractor();

private slots:
    void onActionLoadDataSet();
    void onActionCompute();
    void onActionSave();
    void onActionSaveConfig();

private:
    QScopedPointer<Ui::ToolWindowBoundaryExtractor> m_ui;
    CloudViewer* m_cloudViewer;
    //CloudViewer* m_projectedCloudViewer;
    CloudViewer* m_planeViewer;
    ImageViewer* m_depthViewer;
    ImageViewer* m_depthViewer2;

    QScopedPointer<BoundaryExtractor> m_boundaryExtractor;
    QScopedPointer<DDBPLineExtractor> m_lineExtractor;
    QScopedPointer<SensorReaderDevice> m_device;
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_allBoundary;
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_boundaryPoints;
};

#endif // TOOLWINDOWBOUNDARYEXTRACTOR_H
