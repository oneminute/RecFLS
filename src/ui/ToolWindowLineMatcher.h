#ifndef TOOLWINDOWLINEMATCHER_H
#define TOOLWINDOWLINEMATCHER_H

#include <QMainWindow>
#include <QScopedPointer>

#include "extractor/BoundaryExtractor.h"
#include "extractor/LineExtractor.h"
#include "matcher/LineMatcher.h"
#include "device/SensorReaderDevice.h"
#include "ui/CloudViewer.h"
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
    void showCloudAndLines(CloudViewer* viewer, QList<LineSegment>& lines, boost::shared_ptr<pcl::PointCloud<Line>>& lineCloud);
    void showMatchedClouds();

    void onActionLoadDataSet();
    void onActionMatch();
    void onActionMatchGpu();
    void onActionBeginStep();
    void onActionStep();
    void onActionStepRotationMatch();
    void onActionStepTranslationMatch();
    void onActionReset();

    void onComboBox1CurrentIndexChanged(int index);
    void onActionShowPair(bool isChecked = false);

protected:
    void updateWidgets();

private:
    QScopedPointer<Ui::ToolWindowLineMatcher> m_ui;
    pcl::KdTreeFLANN<Line>::Ptr m_tree;
    CloudViewer* m_cloudViewer1;
    CloudViewer* m_cloudViewer2;
    CloudViewer* m_cloudViewer3;
    QScopedPointer<SensorReaderDevice> m_device;
    QScopedPointer<BoundaryExtractor> m_boundaryExtractor;
    QScopedPointer<LineExtractor> m_lineExtractor;
    QScopedPointer<LineMatcher> m_lineMatcher;

    bool m_isStepMode;
    bool m_isInit;
    bool m_isLoaded;
    int m_iteration;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr m_colorCloud1;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr m_colorCloud2;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_cloud1;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_cloud2;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_filteredCloud1;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_filteredCloud2;
    pcl::PointCloud<Line>::Ptr m_lineCloud1;
    pcl::PointCloud<Line>::Ptr m_lineCloud2;
    QMap<int, int> m_pairs;
    QList<int> m_pairIndices;
    cuda::GpuFrame m_frameGpu1;
    cuda::GpuFrame m_frameGpu2;

    Eigen::Quaternionf m_rotationDelta;
    Eigen::Vector3f m_translationDelta;
    Eigen::Matrix3f m_rotation;
    Eigen::Vector3f m_translation;
    Eigen::Matrix4f m_m;
    float m_rotationError;
    float m_translationError;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif // TOOLWINDOWLINEMATCHER_H
