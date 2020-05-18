#ifndef TOOLWINDOWLINEMATCHER_H
#define TOOLWINDOWLINEMATCHER_H

#include <QMainWindow>
#include <QScopedPointer>

#include "extractor/BoundaryExtractor.h"
#include "extractor/FusedLineExtractor.h"
#include "matcher/LineMatcher.h"
#include "device/SensorReaderDevice.h"
#include "ui/CloudViewer.h"
#include "cuda/FusedLineInternal.h"
#include "cuda/CudaInternal.h"

namespace Ui {
class ToolWindowLineMatcher;
}

class ToolWindowLineMatcher : public QMainWindow
{
    Q_OBJECT

public:
    explicit ToolWindowLineMatcher(QWidget *parent = nullptr);
    ~ToolWindowLineMatcher();

    void initCompute();
    void compute();
    void stepCompute();

protected:
    void showCloudAndLines(CloudViewer* viewer, pcl::PointCloud<LineSegment>::Ptr& lines);
    void showMatchedClouds();

    void onActionLoadDataSet();
    void onActionMatch();
    void onActionMatchGpu();
    void onActionBeginStep();
    void onActionStep();
    void onActionReset();

    void onComboBox1CurrentIndexChanged(int index);
    void onActionShowPair(bool isChecked = false);

protected:
    void updateWidgets();

private:
    QScopedPointer<Ui::ToolWindowLineMatcher> m_ui;
    pcl::KdTreeFLANN<LineSegment>::Ptr m_tree;
    CloudViewer* m_cloudViewer1;
    CloudViewer* m_cloudViewer2;
    CloudViewer* m_cloudViewer3;
    QScopedPointer<SensorReaderDevice> m_device;
    QScopedPointer<BoundaryExtractor> m_boundaryExtractor;
    QScopedPointer<FusedLineExtractor> m_lineExtractor;
    QScopedPointer<LineMatcher> m_lineMatcher;

    bool m_isStepMode;
    bool m_isInit;
    bool m_isLoaded;
    int m_iteration;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr m_colorCloudSrc;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr m_colorCloudDst;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_cloudSrc;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_cloudDst;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_filteredCloudSrc;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_filteredCloudDst;
    pcl::PointCloud<LineSegment>::Ptr m_linesCloudSrc;
    pcl::PointCloud<LineSegment>::Ptr m_linesCloudDst;
    QMap<int, int> m_pairs;
    QList<int> m_pairIndices;
    //cuda::FusedLineFrame m_frameGpuSrc;
    //cuda::FusedLineFrame m_frameGpuDst;
    cuda::GpuFrame m_frameGpuBESrc;
    cuda::GpuFrame m_frameGpuBEDst;

    Eigen::Matrix4f m_pose;
    float m_rotationError;
    float m_translationError;

    //QMap<int, LineSegment> m_linesSrc;
    //QMap<int, LineSegment> m_linesDst;
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_beCloudSrc;
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_beCloudDst;
    QMap<int, pcl::PointCloud<pcl::PointXYZI>::Ptr> m_groupPointsSrc;
    QMap<int, pcl::PointCloud<pcl::PointXYZI>::Ptr> m_groupPointsDst;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif // TOOLWINDOWLINEMATCHER_H
