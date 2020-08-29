#ifndef TOOLWINDOWFUSEDLINEEXTRACTOR_H
#define TOOLWINDOWFUSEDLINEEXTRACTOR_H


#include <QMainWindow>
#include <QScopedPointer>
#include <pcl/search/kdtree.h>

#include "ui/CloudViewer.h"
#include "cuda/FusedLineInternal.h"
#include "cuda/CudaInternal.h"
#include "common/Frame.h"
#include "common/FLFrame.h"

namespace Ui {
    class ToolWindowFusedLineExtractor;
}

class Device;
class FusedLineExtractor;

class ToolWindowFusedLineExtractor : public QMainWindow
{
    Q_OBJECT

public:
    explicit ToolWindowFusedLineExtractor(QWidget *parent = nullptr);
    ~ToolWindowFusedLineExtractor();

    void initCompute();
    void compute();

protected:
    void onActionLoadDataSet();
    void onActionCompute();
    void onActionShowPoints();

protected:
    void updateWidgets();

private:
    QScopedPointer<Ui::ToolWindowFusedLineExtractor> m_ui;
    CloudViewer* m_cloudViewer;
    QScopedPointer<Device> m_device;
    QScopedPointer<FusedLineExtractor> m_extractor;

    bool m_isInit;
    bool m_isLoaded;

    pcl::PointCloud<pcl::PointXYZI>::Ptr m_cloud;
    pcl::PointCloud<pcl::Normal>::Ptr m_normals;

    //cuda::FusedLineFrame m_frameGpu;
    //cuda::GpuFrame m_frameBEGpu;
    Frame m_frame;
    FLFrame m_flFrame;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif // TOOLWINDOWFUSEDLINEEXTRACTOR_H
