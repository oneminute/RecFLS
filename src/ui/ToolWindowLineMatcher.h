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
    void showCloudAndLines(CloudViewer* viewer, QList<Plane>& planes, QList<LineSegment>& lines, boost::shared_ptr<pcl::PointCloud<Line>>& mslCloud);
    void showMatchedClouds();

    void onActionLoadDataSet();
    void onActionMatch();
    void onActionMatchGpu();
    void onActionBeginStep();
    void onActionStepRotationMatch();
    void onActionStepTranslationMatch();
    void onActionReset();

    void onComboBox1CurrentIndexChanged(int index);
    void onActionShowPair(bool isChecked = false);

protected:
    void updateWidgets();

private:
    QScopedPointer<Ui::ToolWindowLineMatcher> m_ui;
    //pcl::KdTreeFLANN<MSLPoint>::Ptr m_tree;
    CloudViewer* m_cloudViewer1;
    CloudViewer* m_cloudViewer2;
    CloudViewer* m_cloudViewer3;
    QScopedPointer<SensorReaderDevice> m_device;
    QScopedPointer<BoundaryExtractor> m_boundaryExtractor;
    QScopedPointer<LineExtractor> m_lineExtractor;
    QScopedPointer<LineMatcher> m_lineMatcher;

    bool m_isStepMode;
    bool m_isInit;
    bool m_useCuda;
    bool m_isLoaded;
    int m_iteration;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr m_colorCloud1;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr m_colorCloud2;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_cloud1;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_cloud2;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_filteredCloud1;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_filteredCloud2;
    pcl::PointCloud<Line>::Ptr m_mslCloud1;
    pcl::PointCloud<Line>::Ptr m_mslCloud2;
    //pcl::PointCloud<MSLPoint>::Ptr m_mslPointCloud1;
    //pcl::PointCloud<MSLPoint>::Ptr m_mslPointCloud2;
    QList<Plane> m_planes1;
    QList<Plane> m_planes2;
    QList<LineChain> m_chains1;
    QList<LineChain> m_chains2;
    pcl::PointCloud<LineDescriptor2>::Ptr m_desc1;
    pcl::PointCloud<LineDescriptor2>::Ptr m_desc2;
    QMap<int, int> m_pairs;
    QList<int> m_pairIndices;
    cuda::GpuFrame m_frameGpu1;
    cuda::GpuFrame m_frameGpu2;

    float m_diameter1;
    float m_diameter2;
    Eigen::Quaternionf m_rotationDelta;
    Eigen::Vector3f m_translationDelta;
    Eigen::Matrix3f m_rotation;
    Eigen::Vector3f m_translation;
    float m_rotationError;
    float m_translationError;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif // TOOLWINDOWLINEMATCHER_H
