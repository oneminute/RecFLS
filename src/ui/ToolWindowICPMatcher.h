#ifndef TOOLWINDOWICPMATCHER_H
#define TOOLWINDOWICPMATCHER_H

#include <QMainWindow>
#include <QScopedPointer>
#include <pcl/search/kdtree.h>

#include "ui/CloudViewer.h"
#include "cuda/IcpInternal.h"

namespace Ui {
class ToolWindowICPMatcher;
}

class SensorReaderDevice;
class ICPMatcher;

class ToolWindowICPMatcher : public QMainWindow
{
    Q_OBJECT

public:
    explicit ToolWindowICPMatcher(QWidget *parent = nullptr);
    ~ToolWindowICPMatcher();

    void initCompute();
    void compute();
    void stepCompute();

protected:
    void showMatchedClouds();

    void onActionLoadDataSet();
    void onActionMatch();
    void onActionComputeGPU();
    void onActionStepReset();
    void onActionStep();
    void onActionStepGPU();
    void onActionReset();

    void onComboBoxFrameSrcCurrentIndexChanged(int index);

protected:
    void updateWidgets();

private:
    QScopedPointer<Ui::ToolWindowICPMatcher> m_ui;
    pcl::search::KdTree<pcl::PointXYZI>::Ptr m_tree;
    CloudViewer* m_cloudViewer;
    QScopedPointer<SensorReaderDevice> m_device;
    QScopedPointer<ICPMatcher> m_icp;

    bool m_isStepMode;
    bool m_isInit;
    bool m_isLoaded;
    int m_iteration;

    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr m_colorCloudSrc;
    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr m_colorCloudDst;
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_cloudSrc;
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_cloudDst;
    pcl::PointCloud<pcl::Normal>::Ptr m_normalsSrc;
    pcl::PointCloud<pcl::Normal>::Ptr m_normalsDst;

    cuda::IcpFrame m_frameSrc;
    cuda::IcpFrame m_frameDst;
    cuda::IcpCache m_cache;

    Eigen::Matrix3f m_rotationDelta;
    Eigen::Vector3f m_translationDelta;
    Eigen::Matrix3f m_rotation;
    Eigen::Vector3f m_translation;
    Eigen::Matrix4f m_pose;
    float m_rotationError;
    float m_translationError;
    int m_pairs = 0;
    float m_error = 0;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif // TOOLWINDOWICPMATCHER_H
