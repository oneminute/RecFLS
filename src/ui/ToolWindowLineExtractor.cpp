#include "ToolWindowLineExtractor.h"
#include "ui_ToolWindowLineExtractor.h"

#include <QDebug>
#include <QDateTime>
#include <QtMath>
#include <QFileDialog>
#include <QDir>

#include "common/Parameters.h"
#include "util/Utils.h"
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

    m_cloudViewer1 = new CloudViewer(this);
    m_cloudViewer2 = new CloudViewer(this);
    m_cloudViewer3 = new CloudViewer(this);

    m_ui->layoutPointCloud->addWidget(m_cloudViewer1);
    m_ui->layoutSecondary->addWidget(m_cloudViewer2);
    m_ui->layoutSecondary->addWidget(m_cloudViewer3);

    m_cloudViewer1->setCameraPosition(0, 0, -1.5f, 0, 0, 0, 0, -1, 0);
    m_cloudViewer2->setCameraPosition(0, 0, 1.5f, 0, 0, 0, 1, 0, 0);

    connect(m_ui->actionLoad_Point_Cloud, &QAction::triggered, this, &ToolWindowLineExtractor::onActionLoadPointCloud);
    connect(m_ui->actionGenerate_Line_Point_Cloud, &QAction::triggered, this, &ToolWindowLineExtractor::onActionGenerateLinePointCloud);
    connect(m_ui->actionParameterized_Points_Analysis, &QAction::triggered, this, &ToolWindowLineExtractor::onActionParameterizedPointsAnalysis);
    connect(m_ui->actionSave_Config, &QAction::triggered, this, &ToolWindowLineExtractor::onActionSaveConfig);
}

ToolWindowLineExtractor::~ToolWindowLineExtractor()
{
}

void ToolWindowLineExtractor::onActionParameterizedPointsAnalysis()
{
    DDBPLineExtractor extractor;
    extractor.setSearchRadius(m_ui->doubleSpinBoxSearchRadius->value());
    extractor.setMinNeighbours(m_ui->spinBoxMinNeighbours->value());
    extractor.setSearchErrorThreshold(m_ui->doubleSpinBoxSearchErrorThreshold->value());
    extractor.setAngleSearchRadius(qDegreesToRadians(m_ui->doubleSpinBoxAngleSearchRadius->value()) * M_1_PI);
    extractor.setAngleMinNeighbours(m_ui->spinBoxAngleMinNeighbours->value());
    extractor.setMappingTolerance(m_ui->doubleSpinBoxClusterTolerance->value());
    extractor.setAngleMappingMethod(m_ui->comboBoxAngleMappingMethod->currentIndex());
    extractor.setMinLineLength(m_ui->doubleSpinBoxMinLineLength->value());
    extractor.setRegionGrowingZDistanceThreshold(m_ui->doubleSpinBoxZDistanceThreshold->value());
    extractor.setMslRadiusSearch(m_ui->doubleSpinBoxMSLRadiusSearch->value());

    QList<LineSegment> lines = extractor.compute(m_cloud);

    pcl::PointCloud<pcl::PointXYZI>::Ptr angleCloud = extractor.angleCloud();
    QList<float> densityList = extractor.densityList();
    QList<int> angleCloudIndices = extractor.angleCloudIndices();
    QMap<int, std::vector<int>> subCloudIndices = extractor.subCloudIndices();
    pcl::PointCloud<pcl::PointXYZI>::Ptr mappingCloud = extractor.mappingCloud();
    QList<float> errors = extractor.errors();
    pcl::PointCloud<pcl::PointXYZI>::Ptr linedCloud = extractor.linedCloud();
    QList<int> linePointsCount = extractor.linePointsCount();
    pcl::PointCloud<MSL>::Ptr mslCloud = extractor.mslCloud();

    m_cloudViewer1->visualizer()->removeAllPointClouds();
    m_cloudViewer1->visualizer()->removeAllShapes();
    m_cloudViewer2->visualizer()->removeAllPointClouds();
    m_cloudViewer2->visualizer()->removeAllShapes();
    m_cloudViewer3->visualizer()->removeAllPointClouds();
    m_cloudViewer3->visualizer()->removeAllShapes();

    pcl::PointCloud<pcl::PointXYZI>::Ptr densityCloud(new pcl::PointCloud<pcl::PointXYZI>);
    for (int i = 0; i < densityList.size(); i++)
    {
        int index = angleCloudIndices[i];
        //qDebug() << indexList[i] << m_density[indexList[i]];

        pcl::PointXYZI ptAngle = angleCloud->points[index];
        ptAngle.intensity = densityList[index];
        densityCloud->push_back(ptAngle);
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr boundaryCloud2(new pcl::PointCloud<pcl::PointXYZI>);
    for (QMap<int, std::vector<int>>::iterator i = subCloudIndices.begin(); i != subCloudIndices.end(); i++)
    {
        int index = i.key();
        pcl::PointXYZI pt = m_cloud->points[index];
        pt.intensity = index;
        int color = qrand();

        for (std::vector<int>::iterator itNeighbour = i.value().begin(); itNeighbour != i.value().end(); itNeighbour++)
        {
            int neighbourIndex = *itNeighbour;
            pcl::PointXYZI ptNeighbour = m_cloud->points[neighbourIndex];
            ptNeighbour.intensity = color;
            boundaryCloud2->push_back(ptNeighbour);
        }
    }

    if (m_ui->radioButtonShowBoundaryCloud->isChecked())
    {
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> iColor(m_cloud, "intensity");
        m_cloudViewer1->visualizer()->addPointCloud(m_cloud, iColor, "original cloud");
        m_cloudViewer1->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "original cloud");
    }
    else if (m_ui->radioButtonShowGroupedCloud->isChecked())
    {
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> iColor(boundaryCloud2, "intensity");
        m_cloudViewer1->visualizer()->addPointCloud(boundaryCloud2, iColor, "grouped cloud");
        m_cloudViewer1->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "grouped cloud");
    }
    else if (m_ui->radioButtonShowLinedCloud->isChecked())
    {
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> iColor(linedCloud, "intensity");
        m_cloudViewer1->visualizer()->addPointCloud(linedCloud, iColor, "lined cloud");
        m_cloudViewer1->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "lined cloud");
    }

    if (m_ui->radioButtonShowAngleCloud->isChecked())
    {
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> iColor(angleCloud, "intensity");
        m_cloudViewer2->visualizer()->addPointCloud(angleCloud, iColor, "angle cloud");
        m_cloudViewer2->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "angle cloud");
    }
    else if (m_ui->radioButtonShowDensityCloud->isChecked())
    {
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> iColor(densityCloud, "intensity");
        m_cloudViewer2->visualizer()->addPointCloud(densityCloud, iColor, "density cloud");
        m_cloudViewer2->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "density cloud");
    }

    {
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> iColor(mappingCloud, "intensity");
        m_cloudViewer3->visualizer()->addPointCloud(mappingCloud, iColor, "mapping cloud");
        m_cloudViewer3->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "mapping cloud");

        for (int i = 0; i < lines.size()/* && errors[i]*/; i++)
        {
            double r = rand() * 1.0 / RAND_MAX;
            double g = rand() * 1.0 / RAND_MAX;
            double b = rand() * 1.0 / RAND_MAX;
            LineSegment line = lines[i];
            std::string lineNo = "line_" + std::to_string(i);
            std::string textNo = "text_" + std::to_string(i);
            qDebug() << QString::fromStdString(lineNo) << line.length() << errors[i] << linePointsCount[i];
            pcl::PointXYZI start, end, middle;
            start.getVector3fMap() = line.start();
            end.getVector3fMap() = line.end();
            Eigen::Vector3f dir = line.direction();
            middle.getVector3fMap() = line.middle();
            //m_cloudViewer->visualizer()->addArrow(end, start, r, g, b, 0, lineNo);
            m_cloudViewer1->visualizer()->addLine(start, end, r, g, b, lineNo);
            m_cloudViewer1->visualizer()->addText3D(std::to_string(i), middle, 0.025, 1, 1, 1, textNo);
            m_cloudViewer1->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 4, lineNo);
        }
    }

    {
        for (int i = 0; i < mslCloud->size(); i++)
        {
            MSL msl = mslCloud->points[i];
            pcl::PointXYZI start, end;
            start.getVector3fMap() = msl.getEndPoint(-3);
            end.getVector3fMap() = msl.getEndPoint(3);
            QString lineName = QString("msl_%1").arg(i);
            m_cloudViewer1->visualizer()->addLine(start, end, 255, 0, 0, lineName.toStdString());
            m_cloudViewer1->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, lineName.toStdString());
        }
    }
}

void ToolWindowLineExtractor::onActionSaveConfig()
{
    PARAMETERS.setValue("search_radius", m_ui->doubleSpinBoxSearchRadius->value(), "LineExtractor");
    PARAMETERS.setValue("min_neighbours", m_ui->spinBoxMinNeighbours->value(), "LineExtractor");
    PARAMETERS.setValue("search_error_threshold", m_ui->doubleSpinBoxSearchErrorThreshold->value(), "LineExtractor");
    PARAMETERS.setValue("angle_search_radius", qDegreesToRadians(m_ui->doubleSpinBoxAngleSearchRadius->value()) * M_1_PI, "LineExtractor");
    PARAMETERS.setValue("angle_min_neighbours", m_ui->spinBoxAngleMinNeighbours->value(), "LineExtractor");
    PARAMETERS.setValue("mapping_tolerance", m_ui->doubleSpinBoxClusterTolerance->value(), "LineExtractor");
    PARAMETERS.setValue("angle_mapping_method", m_ui->comboBoxAngleMappingMethod->currentIndex(), "LineExtractor");
    PARAMETERS.setValue("min_line_length", m_ui->doubleSpinBoxMinLineLength->value(), "LineExtractor");
    PARAMETERS.setValue("region_growing_z_distance_threshold", m_ui->doubleSpinBoxZDistanceThreshold->value(), "LineExtractor");
    PARAMETERS.setValue("msl_radius_search", m_ui->doubleSpinBoxMSLRadiusSearch->value(), "LineExtractor");

    PARAMETERS.save();
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
    
    for (int i = 0; i < cloud->size(); i++)
    {
        pcl::PointXYZ inPt = cloud->points[i];
        pcl::PointXYZI outPt;
        float offsetX = 0;
        float offsetY = 0;
        float offsetZ = 0;
        if (m_ui->checkBoxEnableRandom->isChecked())
        {
            offsetX = qrand() * 1.f / RAND_MAX * m_ui->doubleSpinBoxRandomOffset->value();
            offsetY = qrand() * 1.f / RAND_MAX * m_ui->doubleSpinBoxRandomOffset->value();
            offsetZ = qrand() * 1.f / RAND_MAX * m_ui->doubleSpinBoxRandomOffset->value();
        }
        outPt.x = inPt.x + offsetX;
        outPt.y = inPt.y + offsetY;
        outPt.z = inPt.z + offsetZ;
        outPt.intensity = i;
        m_cloud->push_back(outPt);
    }

    m_cloudViewer1->visualizer()->removeAllPointClouds();
    m_cloudViewer1->visualizer()->removeAllShapes();
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> iColor(m_cloud, "intensity");
    m_cloudViewer1->visualizer()->addPointCloud(m_cloud, iColor, "original cloud");
    m_cloudViewer1->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original cloud");
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

    m_cloudViewer1->visualizer()->removeAllPointClouds();
    m_cloudViewer1->visualizer()->removeAllShapes();

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
    m_cloudViewer1->visualizer()->addPointCloud(m_cloud, iColor, "original cloud");
    m_cloudViewer1->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original cloud");
}
