#ifndef TOOLWINDOWBOUNDARYEXTRACTOR_H
#define TOOLWINDOWBOUNDARYEXTRACTOR_H

#include <QMainWindow>
#include <QScopedPointer>

#include "ui/CloudViewer.h"
#include "ui/ImageViewer.h"
#include "extractor/BoundaryExtractor.h"
#include "extractor/LineExtractor.h"
#include "device/SensorReaderDevice.h"
#include "cuda/CudaInternal.h"

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
    void init();
    void onActionLoadDataSet();
    //void onActionCompute();
    void onActionComputeGPU();
    void onActionComputeVBRG();
    void onActionSave();
    void onActionSaveConfig();

    void initDebugPixels(Frame& frame);

private:
    QScopedPointer<Ui::ToolWindowBoundaryExtractor> m_ui;
    CloudViewer* m_cloudViewer;
    CloudViewer* m_projectedCloudViewer;
    //CloudViewer* m_planeViewer;
    //ImageViewer* m_depthViewer;
    ImageViewer* m_depthViewer2;

    QScopedPointer<BoundaryExtractor> m_boundaryExtractor;
    QScopedPointer<LineExtractor> m_lineExtractor;
    QScopedPointer<SensorReaderDevice> m_device;
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_allBoundary;
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_boundaryPoints;

    cuda::GpuFrame m_frameGpu;
    bool m_init;
};

#endif // TOOLWINDOWBOUNDARYEXTRACTOR_H
