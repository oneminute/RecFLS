#include "ToolWindowLineExtractor.h"
#include "ui_ToolWindowLineExtractor.h"

#include <QDebug>
#include <QDateTime>
#include <QtMath>

#include "common/Parameters.h"
#include "util/Utils.h"
#include "extractor/LineExtractor.hpp"

#include <pcl/visualization/pcl_visualizer.h>

ToolWindowLineExtractor::ToolWindowLineExtractor(QWidget* parent)
    : QMainWindow(parent)
    , m_ui(new Ui::ToolWindowLineExtractor)
{
    m_ui->setupUi(this);

    m_cloudViewer = new CloudViewer(this);
    m_cloudViewerSecondary = new CloudViewer(this);
    //m_cloudViewerSecondary->removeAllCoordinates();

    m_ui->layoutPointCloud->addWidget(m_cloudViewer);
    m_ui->layoutSecondary->addWidget(m_cloudViewerSecondary);

    connect(m_ui->actionGenerate_Line_Point_Cloud, &QAction::triggered, this, &ToolWindowLineExtractor::onActionGenerateLinePointCloud);
    connect(m_ui->actionAnalysis, &QAction::triggered, this, &ToolWindowLineExtractor::onActionAnalysis);
}

ToolWindowLineExtractor::~ToolWindowLineExtractor()
{
}

void ToolWindowLineExtractor::onActionAnalysis()
{
    if (!m_extractor)
    {
        m_extractor.reset(new LineExtractor<pcl::PointXYZI, pcl::PointXYZI>(
            PARAMETERS.floatValue("segment_distance_threshold", 0.1f, "LineExtractor"),
            PARAMETERS.intValue("min_line_points", 9, "LineExtractor"),
            PARAMETERS.floatValue("pca_error_threshold", 0.005f, "LineExtractor"),
            PARAMETERS.floatValue("line_cluster_angle_threshold", 20.0f, "LineExtractor"),
            PARAMETERS.floatValue("lines_distance_threshold", 0.01f, "LineExtractor"),
            PARAMETERS.floatValue("lines_chain_distance_threshold", 0.01f, "LineExtractor")
        ));
    }

    pcl::PointCloud<pcl::PointXYZI> leCloud;
    m_extractor->compute(*m_cloud, leCloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr dirCloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr clusterCloud = m_extractor->parameterizedPointMappingCluster(dirCloud);
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> iColor(clusterCloud, "intensity");
    m_cloudViewerSecondary->visualizer()->addPointCloud(clusterCloud, iColor, "cluster points");
    m_cloudViewerSecondary->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cluster points");

    qDebug() << clusterCloud->size();
    for (int i = 0; i < clusterCloud->size(); i++)
    {
        pcl::PointXYZI pt = clusterCloud->points[i];

        Eigen::Vector3f dir = dirCloud->points[i].getVector3fMap();
        if (i % 100 == 0)
        {
            pcl::PointXYZI ptStart = m_cloud->points[i];
            pcl::PointXYZI ptEnd;
            ptEnd.getVector3fMap() = ptStart.getVector3fMap() + dir * 0.01f;
            QString arrowId = QString("arrow_%1").arg(i);
            m_cloudViewer->visualizer()->addArrow(ptStart, ptEnd, 1, 1, 1, 0, arrowId.toStdString());
        }

        //dir = dir.transpose();
        //qDebug() << fixed << qSetRealNumberPrecision(2) << qSetFieldWidth(6) << qRadiansToDegrees(pt.x * M_PI) << qRadiansToDegrees(pt.y / 2 * M_PI) << pt.z << dir;
    }
}

void ToolWindowLineExtractor::onActionGenerateLinePointCloud()
{
    m_cloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    Eigen::Vector3f start1(0.0f, 0.f, 0.1f);
    Eigen::Vector3f start2(0.0f, 0.f, 0.1f);
    Eigen::Vector3f start3(0.3f, 0.f, -0.1f);
    Eigen::Vector3f dir1(0, 1.f, 0.f);
    Eigen::Vector3f dir2(1, 0.f, 0.f);
    Eigen::Vector3f dir3(1, 1.f, 1.f);
    dir1.normalize();
    dir2.normalize();
    dir3.normalize();

    qsrand(QDateTime::currentMSecsSinceEpoch());

    for (int i = 0; i < 3000; i++)
    {
        if (i < 1000)
        {
            Eigen::Vector3f ePt = start1 + dir1 * i * 0.001f;
            pcl::PointXYZI point;
            float offsetX = qrand() * 1.f / RAND_MAX * 0.00025;
            float offsetY = qrand() * 1.f / RAND_MAX * 0.00025;
            float offsetZ = qrand() * 1.f / RAND_MAX * 0.00025;
            point.x = ePt.x() + offsetX;
            point.y = ePt.y() + offsetY;
            point.z = ePt.z() + offsetZ;
            point.intensity = i;
            m_cloud->push_back(point);
        }
        else if (i >= 1000 && i < 2000)
        {
            continue;
            Eigen::Vector3f ePt = start2 + dir2 *(i - 1000) * 0.001f;
            pcl::PointXYZI point;
            float offsetX = qrand() * 1.f / RAND_MAX * 0.00025;
            float offsetY = qrand() * 1.f / RAND_MAX * 0.00025;
            float offsetZ = qrand() * 1.f / RAND_MAX * 0.00025;
            point.x = ePt.x() + offsetX;
            point.y = ePt.y() + offsetY;
            point.z = ePt.z() + offsetZ;
            point.intensity = i;
            m_cloud->push_back(point);
        }
        else if (i >= 2000)
        {
            Eigen::Vector3f ePt = start3 + dir3 *(i - 2000) * 0.001f;
            pcl::PointXYZI point;
            float offsetX = qrand() * 1.f / RAND_MAX * 0.00025;
            float offsetY = qrand() * 1.f / RAND_MAX * 0.00025;
            float offsetZ = qrand() * 1.f / RAND_MAX * 0.00025;
            point.x = ePt.x() + offsetX;
            point.y = ePt.y() + offsetY;
            point.z = ePt.z() + offsetZ;
            point.intensity = i;
            m_cloud->push_back(point);
        }
    }
    m_cloud->is_dense = true;

    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> iColor(m_cloud, "intensity");
    m_cloudViewer->visualizer()->addPointCloud(m_cloud, iColor, "original cloud");
    m_cloudViewer->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original cloud");
}
