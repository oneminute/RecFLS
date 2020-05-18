#ifndef TOOLWINDOWLINEEXTRACTOR_H
#define TOOLWINDOWLINEEXTRACTOR_H

#include <QMainWindow>
#include <QScopedPointer>

#include <pcl/common/common.h>
#include "CloudViewer.h"

#include "extractor/BoundaryExtractor.h"
#include "extractor/FusedLineExtractor.h"
#include "extractor/LineSegment.h"
#include "device/SensorReaderDevice.h"
#include "cuda/CudaInternal.h"

namespace Ui {
class ToolWindowLineExtractor;
}

class ToolWindowLineExtractor : public QMainWindow
{
    Q_OBJECT

public:
    explicit ToolWindowLineExtractor(QWidget *parent = nullptr);
    ~ToolWindowLineExtractor() override;

    void showLines();

private slots:
    void init();
    void compute();
    void onActionLoadDataSet();
    void onActionLoadPointCloud();
    void onActionParameterizedPointsAnalysis();
    void onActionComputeGPU();
    void onActionShowLineChain(bool checked = false);

private:
    QScopedPointer<Ui::ToolWindowLineExtractor> m_ui;
    QScopedPointer<SensorReaderDevice> m_device;
    QScopedPointer<FusedLineExtractor> m_lineExtractor;
    QScopedPointer<BoundaryExtractor> m_boundaryExtractor;

    CloudViewer *m_cloudViewer1;
    //CloudViewer *m_cloudViewer2;
    //CloudViewer *m_cloudViewer3;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_originalCloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_dataCloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr m_colorCloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_filteredCloud;
    QList<Plane> m_planes;
    QList<LineSegment> m_lines;
    pcl::PointCloud<LineSegment>::Ptr m_lineCloud;
    cuda::GpuFrame m_frameGpu;

    bool m_fromDataSet;
    bool m_useCuda;
    bool m_init;
};

#endif  // TOOLWINDOWLINEEXTRACTOR_H
