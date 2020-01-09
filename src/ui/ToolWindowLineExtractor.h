#ifndef TOOLWINDOWLINEEXTRACTOR_H
#define TOOLWINDOWLINEEXTRACTOR_H

#include <QMainWindow>
#include <QScopedPointer>

#include <pcl/common/common.h>
#include "CloudViewer.h"
#include "extractor/LineExtractor.h"

namespace Ui {
class ToolWindowLineExtractor;
}

class ToolWindowLineExtractor : public QMainWindow
{
    Q_OBJECT

public:
    explicit ToolWindowLineExtractor(QWidget *parent = nullptr);
    ~ToolWindowLineExtractor() override;

private slots:
    void onActionGenerateLinePointCloud();
    void onActionAnalysis();

private:
    QScopedPointer<Ui::ToolWindowLineExtractor> m_ui;

    QScopedPointer<LineExtractor<pcl::PointXYZI, pcl::PointXYZI>> m_extractor;
    CloudViewer *m_cloudViewer;
    CloudViewer *m_cloudViewerSecondary;
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_cloud;
};

#endif  // TOOLWINDOWLINEEXTRACTOR_H
