#include "ToolWindowLineExtractor.h"
#include "ui_ToolWindowLineExtractor.h"

#include <QDebug>
#include <QDateTime>
#include <QtMath>
#include <QFileDialog>
#include <QDir>

#include "common/Parameters.h"
#include "util/Utils.h"
#include "extractor/LineExtractor.hpp"
#include "extractor/DDBPLineExtractor.h"

#include <pcl/io/ply_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/pcd_io.h>
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

    connect(m_ui->actionLoad_Point_Cloud, &QAction::triggered, this, &ToolWindowLineExtractor::onActionLoadPointCloud);
    connect(m_ui->actionGenerate_Line_Point_Cloud, &QAction::triggered, this, &ToolWindowLineExtractor::onActionGenerateLinePointCloud);
    connect(m_ui->actionParameterized_Points_Analysis, &QAction::triggered, this, &ToolWindowLineExtractor::onActionParameterizedPointsAnalysis);
}

ToolWindowLineExtractor::~ToolWindowLineExtractor()
{
}

void ToolWindowLineExtractor::onActionParameterizedPointsAnalysis()
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

    m_cloudViewerSecondary->visualizer()->removeAllPointClouds();
    m_cloudViewerSecondary->visualizer()->removeAllShapes();

    //pcl::PointCloud<pcl::PointXYZI> leCloud;
    //m_extractor->compute(*m_cloud, leCloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr dirCloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr clusterCloud = m_extractor->parameterizedPointMappingCluster(m_cloud, dirCloud);
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> iColor(clusterCloud, "intensity");
    m_cloudViewerSecondary->visualizer()->addPointCloud(clusterCloud, iColor, "cluster points");
    m_cloudViewerSecondary->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cluster points");

    pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
    std::vector<pcl::PointIndices> clusterIndices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
    ec.setClusterTolerance(m_ui->doubleSpinBoxClusterTolerance->value());
    ec.setMinClusterSize(10);
    ec.setMaxClusterSize(clusterCloud->size());
    ec.setSearchMethod(tree);
    ec.setInputCloud(clusterCloud);
    ec.extract(clusterIndices);

    int count = 0;
    for (std::vector<pcl::PointIndices>::iterator i = clusterIndices.begin(); i != clusterIndices.end(); i++, count++)
    {
        QString clusterId = QString("cluster_%1").arg(count);
        QString lineId = QString("line_%1").arg(count);
        QString arrowId = QString("arrow_%1").arg(count);
        qDebug() << clusterId << ". cluster size:" << i->indices.size();
        Eigen::Vector3f center(0, 0, 0);
        Eigen::Vector3f dir(0, 0, 0);
        Eigen::Vector3f mean(0, 0, 0);
        for (std::vector<int>::iterator itInd = i->indices.begin(); itInd != i->indices.end(); itInd++)
        {
            int index = *itInd;
            Eigen::Vector3f point = clusterCloud->points[index].getVector3fMap();
            center += point;
            //clusterCloud->points[*itInd].intensity = count;
            Eigen::Vector3f ptDir = dirCloud->points[clusterCloud->points[index].intensity].getVector3fMap();
            dir += ptDir;
            Eigen::Vector3f pt = m_cloud->points[clusterCloud->points[index].intensity].getVector3fMap();
            mean += pt;
        }
        dir = dir / i->indices.size();
        dir.normalize();
        mean = mean / i->indices.size();

        qDebug() << dir.x() << dir.y() << dir.z() << mean.x() << mean.y() << mean.z();

        Eigen::Vector3f start = m_cloud->points[clusterCloud->points[i->indices[0]].intensity].getVector3fMap();
        Eigen::Vector3f end = start;
        for (std::vector<int>::iterator itInd = i->indices.begin(); itInd != i->indices.end(); itInd++)
        {
            int index = clusterCloud->points[*itInd].intensity;
            Eigen::Vector3f point = m_cloud->points[index].getVector3fMap();
            Eigen::Vector3f projPt = closedPointOnLine(point, dir, mean);
            if ((start - projPt).dot(dir) > 0)
            {
                start = projPt;
            }
            if ((projPt - end).dot(dir) > 0)
            {
                end = projPt;
            }
        }

        pcl::PointXYZ startPt, endPt;
        startPt.getVector3fMap() = start;
        endPt.getVector3fMap() = end;
        float length = (end - start).norm();
        qDebug() << start.x() << start.y() << start.z() << end.x() << end.y() << end.z() << length;
        m_cloudViewer->visualizer()->addLine(startPt, endPt, 1, 1, 0, lineId.toStdString());
        //m_cloudViewer->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 4, lineId.toStdString());
        //m_cloudViewer->visualizer()->addArrow(startPt, endPt, 1, 1, 0, 0, arrowId.toStdString());
        m_cloudViewer->visualizer()->addLine(startPt, endPt, 1, 0, 0, lineId.toStdString());

        center = center / i->indices.size();
        pcl::PointXYZ ptCenter;
        ptCenter.getVector3fMap() = center;
        //m_cloudViewerSecondary->visualizer()->addSphere(ptCenter, 0.1 * i->indices.size() / m_cloud->size(), id.toStdString());
        m_cloudViewerSecondary->visualizer()->addText3D(std::to_string(i->indices.size()), ptCenter, 0.01, 1, 1, 1, clusterId.toStdString());
    }

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
            //m_cloudViewer->visualizer()->addArrow(ptStart, ptEnd, 1, 1, 1, 0, arrowId.toStdString());
        }

        //dir = dir.transpose();
        //qDebug() << fixed << qSetRealNumberPrecision(2) << qSetFieldWidth(6) << qRadiansToDegrees(pt.x * M_PI) << qRadiansToDegrees(pt.y / 2 * M_PI) << pt.z << dir;
    }
}

void ToolWindowLineExtractor::onActionLoadPointCloud()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Point Cloud"), QDir::current().absolutePath(), tr("Polygon Files (*.obj *.ply *.pcd)"));
    if (fileName.isEmpty())
    {
        return;
    }
    QFileInfo info(fileName);

    m_cloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    if (info.completeSuffix().contains("obj"))
    {
        pcl::io::loadOBJFile<pcl::PointXYZ>(fileName.toStdString(), *cloud);
    }
    else if (info.completeSuffix().contains("ply"))
    {
        pcl::io::loadPLYFile<pcl::PointXYZ>(fileName.toStdString(), *cloud);
    }
    else if (info.completeSuffix().contains("pcd"))
    {
        pcl::io::loadPCDFile<pcl::PointXYZ>(fileName.toStdString(), *cloud);
    }
    qDebug() << fileName << cloud->size();

    qsrand(QDateTime::currentMSecsSinceEpoch());
    if (m_ui->checkBoxEnableRandom->isChecked())
    {
        for (int i = 0; i < cloud->size(); i++)
        {
            pcl::PointXYZ inPt = cloud->points[i];
            pcl::PointXYZI outPt;
            float offsetX = qrand() * 1.f / RAND_MAX * m_ui->doubleSpinBoxRandomOffset->value();
            float offsetY = qrand() * 1.f / RAND_MAX * m_ui->doubleSpinBoxRandomOffset->value();
            float offsetZ = qrand() * 1.f / RAND_MAX * m_ui->doubleSpinBoxRandomOffset->value();
            outPt.x = inPt.x + offsetX;
            outPt.y = inPt.y + offsetY;
            outPt.z = inPt.z + offsetZ;
            outPt.intensity = i;
            m_cloud->push_back(outPt);
        }
    }

    m_cloudViewer->visualizer()->removeAllPointClouds();
    m_cloudViewer->visualizer()->removeAllShapes();
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> iColor(m_cloud, "intensity");
    m_cloudViewer->visualizer()->addPointCloud(m_cloud, iColor, "original cloud");
    m_cloudViewer->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original cloud");
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

    m_cloudViewer->visualizer()->removeAllPointClouds();
    m_cloudViewer->visualizer()->removeAllShapes();

    qsrand(QDateTime::currentMSecsSinceEpoch());

    for (int i = 0; i < 3000; i++)
    {
        if (i < 1000)
        {
            Eigen::Vector3f ePt = start1 + dir1 * i * 0.001f;
            pcl::PointXYZI point;
            float offsetX = qrand() * 1.f / RAND_MAX * m_ui->doubleSpinBoxRandomOffset->value();
            float offsetY = qrand() * 1.f / RAND_MAX * m_ui->doubleSpinBoxRandomOffset->value();
            float offsetZ = qrand() * 1.f / RAND_MAX * m_ui->doubleSpinBoxRandomOffset->value();
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
            float offsetX = qrand() * 1.f / RAND_MAX * m_ui->doubleSpinBoxRandomOffset->value();
            float offsetY = qrand() * 1.f / RAND_MAX * m_ui->doubleSpinBoxRandomOffset->value();
            float offsetZ = qrand() * 1.f / RAND_MAX * m_ui->doubleSpinBoxRandomOffset->value();
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
            float offsetX = qrand() * 1.f / RAND_MAX * m_ui->doubleSpinBoxRandomOffset->value();
            float offsetY = qrand() * 1.f / RAND_MAX * m_ui->doubleSpinBoxRandomOffset->value();
            float offsetZ = qrand() * 1.f / RAND_MAX * m_ui->doubleSpinBoxRandomOffset->value();
            point.x = ePt.x() + offsetX;
            point.y = ePt.y() + offsetY;
            point.z = ePt.z() + offsetZ;
            point.intensity = i;
            m_cloud->push_back(point);
        }
    }
    m_cloud->is_dense = true;
    qDebug() << "cloud size:" << m_cloud->size();

    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> iColor(m_cloud, "intensity");
    m_cloudViewer->visualizer()->addPointCloud(m_cloud, iColor, "original cloud");
    m_cloudViewer->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original cloud");
}
