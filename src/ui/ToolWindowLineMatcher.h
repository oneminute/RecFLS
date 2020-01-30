#ifndef TOOLWINDOWLINEMATCHER_H
#define TOOLWINDOWLINEMATCHER_H

#include <QMainWindow>
#include <QScopedPointer>

#include "extractor/BoundaryExtractor.h"
#include "extractor/DDBPLineExtractor.h"
#include "matcher/DDBPLineMatcher.h"
#include "device/SensorReaderDevice.h"
#include "ui/CloudViewer.h"

namespace Ui {
class ToolWindowLineMatcher;
}

class ToolWindowLineMatcher : public QMainWindow
{
    Q_OBJECT

public:
    explicit ToolWindowLineMatcher(QWidget *parent = nullptr);
    ~ToolWindowLineMatcher();

    void compute();
    void stepCompute();

protected:
    void ShowCloudAndLines(CloudViewer* viewer, QList<LineSegment>& lines, boost::shared_ptr<pcl::PointCloud<DDBPLineExtractor::MSL>>& mslCloud);

    void onActionLoadDataSet();
    void onActionMatch();


private:
    QScopedPointer<Ui::ToolWindowLineMatcher> m_ui;
    CloudViewer* m_cloudViewer1;
    CloudViewer* m_cloudViewer2;
    CloudViewer* m_cloudViewer3;
    QScopedPointer<SensorReaderDevice> m_device;
    QScopedPointer<BoundaryExtractor> m_boundaryExtractor;
    QScopedPointer<DDBPLineExtractor> m_lineExtractor;
    QScopedPointer<DDBPLineMatcher> m_lineMatcher;
};

#endif // TOOLWINDOWLINEMATCHER_H
