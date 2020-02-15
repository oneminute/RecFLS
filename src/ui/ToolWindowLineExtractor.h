#ifndef TOOLWINDOWLINEEXTRACTOR_H
#define TOOLWINDOWLINEEXTRACTOR_H

#include <QMainWindow>
#include <QScopedPointer>

#include <pcl/common/common.h>
#include "CloudViewer.h"

#include "extractor/BoundaryExtractor.h"
#include "extractor/LineExtractor.h"
#include "device/SensorReaderDevice.h"

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
    void onActionLoadDataSet();
    void onActionLoadPointCloud();
    void onActionGenerateLinePointCloud();
    void onActionParameterizedPointsAnalysis();
    void onActionSaveConfig();
    void onActionShowLineChain(bool checked = false);

private:
    QScopedPointer<Ui::ToolWindowLineExtractor> m_ui;
    QScopedPointer<SensorReaderDevice> m_device;
    QScopedPointer<BoundaryExtractor> m_boundaryExtractor;

    CloudViewer *m_cloudViewer1;
    CloudViewer *m_cloudViewer2;
    CloudViewer *m_cloudViewer3;
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_cloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_dataCloud;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr m_colorCloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_filteredCloud;
    QList<Plane> m_planes;
    QList<LineChain> m_chains;
    QList<LineSegment> m_lines;
    pcl::PointCloud<MSL>::Ptr m_mslCloud;

    bool m_fromDataSet;
};

#endif  // TOOLWINDOWLINEEXTRACTOR_H
