#include "ToolWindowLineMatcher.h"
#include "ui_ToolWindowLineMatcher.h"

#include <QDebug>
#include <QtMath>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/pca.h>

#include "common/Parameters.h"

ToolWindowLineMatcher::ToolWindowLineMatcher(QWidget *parent) 
    : QMainWindow(parent)
    , m_ui(new Ui::ToolWindowLineMatcher)
    , m_isStepMode(false)
    , m_isInit(false)
    , m_isLoaded(false)
    , m_iteration(0)
    , m_diameter1(0)
    , m_diameter2(0)
    , m_rotation(Eigen::Quaternionf::Identity())
    , m_translation(Eigen::Vector3f::Zero())
    , m_rotationError(0)
    , m_translationError(0)
{
    m_ui->setupUi(this);

    m_cloudViewer1 = new CloudViewer;
    m_cloudViewer2 = new CloudViewer;
    m_cloudViewer3 = new CloudViewer;

    m_cloudViewer1->setCameraPosition(0, 0, -1.5f, 0, 0, 0, 0, -1, 0);
    m_cloudViewer2->setCameraPosition(0, 0, -1.5f, 0, 0, 0, 0, -1, 0);
    m_cloudViewer3->setCameraPosition(0, 0, -1.5f, 0, 0, 0, 0, -1, 0);

    m_ui->verticalLayoutCloud1->addWidget(m_cloudViewer1);
    m_ui->verticalLayoutCloud2->addWidget(m_cloudViewer2);
    m_ui->verticalLayoutCloud3->addWidget(m_cloudViewer3);

    connect(m_ui->actionLoad_Data_Set, &QAction::triggered, this, &ToolWindowLineMatcher::onActionLoadDataSet);
    connect(m_ui->actionMatch, &QAction::triggered, this, &ToolWindowLineMatcher::onActionMatch);
    connect(m_ui->actionBegin_Step, &QAction::triggered, this, &ToolWindowLineMatcher::onActionBeginStep);
    connect(m_ui->actionStep_Rotaion_Match, &QAction::triggered, this, &ToolWindowLineMatcher::onActionStepRotaionMatch);
    connect(m_ui->actionStep_Translate_Match, &QAction::triggered, this, &ToolWindowLineMatcher::onActionStepTranslationMatch);
    connect(m_ui->actionReset, &QAction::triggered, this, &ToolWindowLineMatcher::onActionReset);

    updateWidgets();
}

ToolWindowLineMatcher::~ToolWindowLineMatcher()
{
}

void ToolWindowLineMatcher::initCompute()
{
    if (!m_boundaryExtractor)
    {
        m_boundaryExtractor.reset(new BoundaryExtractor);
    }

    if (!m_lineExtractor)
    {
        m_lineExtractor.reset(new DDBPLineExtractor);
    }

    if (!m_lineMatcher)
    {
        m_lineMatcher.reset(new DDBPLineMatcher);
    }

    m_boundaryExtractor->setDownsamplingMethod(PARAMETERS.intValue("downsampling_method", 0, "BoundaryExtractor"));
    m_boundaryExtractor->setEnableRemovalFilter(PARAMETERS.boolValue("enable_removal_filter", false, "BoundaryExtractor"));
    m_boundaryExtractor->setDownsampleLeafSize(PARAMETERS.floatValue("downsample_leaf_size", 0.0075f, "BoundaryExtractor"));
    m_boundaryExtractor->setOutlierRemovalMeanK(PARAMETERS.floatValue("outlier_removal_mean_k", 20.f, "BoundaryExtractor"));
    m_boundaryExtractor->setStddevMulThresh(PARAMETERS.floatValue("std_dev_mul_thresh", 1.f, "BoundaryExtractor"));
    m_boundaryExtractor->setGaussianSigma(PARAMETERS.floatValue("gaussian_sigma", 4.f, "BoundaryExtractor"));
    m_boundaryExtractor->setGaussianRSigma(PARAMETERS.floatValue("gaussian_r_sigma", 4.f, "BoundaryExtractor"));
    m_boundaryExtractor->setGaussianRadiusSearch(PARAMETERS.floatValue("gaussian_radius_search", 0.05f, "BoundaryExtractor"));
    m_boundaryExtractor->setNormalsRadiusSearch(PARAMETERS.floatValue("normals_radius_search", 0.05f, "BoundaryExtractor"));
    m_boundaryExtractor->setBoundaryRadiusSearch(PARAMETERS.floatValue("boundary_radius_search", 0.1f, "BoundaryExtractor"));
    m_boundaryExtractor->setBoundaryAngleThreshold(PARAMETERS.floatValue("boundary_angle_threshold", M_PI_2, "BoundaryExtractor"));
    m_boundaryExtractor->setBorderLeft(PARAMETERS.floatValue("border_left", 26, "BoundaryExtractor"));
    m_boundaryExtractor->setBorderRight(PARAMETERS.floatValue("border_right", 22, "BoundaryExtractor"));
    m_boundaryExtractor->setBorderTop(PARAMETERS.floatValue("border_top", 16, "BoundaryExtractor"));
    m_boundaryExtractor->setBorderBottom(PARAMETERS.floatValue("border_bottom", 16, "BoundaryExtractor"));
    m_boundaryExtractor->setProjectedRadiusSearch(qDegreesToRadians(PARAMETERS.floatValue("projected_radius_search", 5, "BoundaryExtractor")));
    m_boundaryExtractor->setVeilDistanceThreshold(PARAMETERS.floatValue("veil_distance_threshold", 0.1f, "BoundaryExtractor"));

    m_lineExtractor->setSearchRadius(PARAMETERS.floatValue("search_radius", 0.1f, "LineExtractor"));
    m_lineExtractor->setMinNeighbours(PARAMETERS.intValue("min_neighbours", 3, "LineExtractor"));
    m_lineExtractor->setSearchErrorThreshold(PARAMETERS.floatValue("search_error_threshold", 0.05f, "LineExtractor"));
    m_lineExtractor->setAngleSearchRadius(PARAMETERS.floatValue("angle_search_radius", M_PI_4, "LineExtractor"));
    m_lineExtractor->setAngleMinNeighbours(PARAMETERS.intValue("angle_min_neighbours", 10, "LineExtractor"));
    m_lineExtractor->setMappingTolerance(PARAMETERS.floatValue("mapping_tolerance", 0.01f, "LineExtractor"));
    m_lineExtractor->setAngleMappingMethod(PARAMETERS.intValue("angle_mapping_method", 0, "LineExtractor"));
    m_lineExtractor->setMinLineLength(PARAMETERS.floatValue("min_line_length", 0.05f, "LineExtractor"));
    m_lineExtractor->setRegionGrowingZDistanceThreshold(PARAMETERS.floatValue("region_growing_z_distance_threshold", 0.005f, "LineExtractor"));
    m_lineExtractor->setMslRadiusSearch(PARAMETERS.floatValue("msl_radius_search", 0.01f, "LineExtractor"));

    int frameIndex1 = m_ui->comboBoxFirstFrame->currentIndex();
    Frame frame1 = m_device->getFrame(frameIndex1);
    int frameIndex2 = m_ui->comboBoxSecondFrame->currentIndex();
    Frame frame2 = m_device->getFrame(frameIndex2);

    pcl::IndicesPtr indices1(new pcl::Indices);
    pcl::IndicesPtr indices2(new pcl::Indices);
    m_cloud1.reset(new pcl::PointCloud<pcl::PointXYZ>);
    m_cloud2.reset(new pcl::PointCloud<pcl::PointXYZ>);

    m_colorCloud1 = frame1.getCloud(*indices1);
    m_colorCloud2 = frame2.getCloud(*indices2);
    pcl::copyPointCloud<pcl::PointXYZRGB, pcl::PointXYZ>(*m_colorCloud1, *m_cloud1);
    pcl::copyPointCloud<pcl::PointXYZRGB, pcl::PointXYZ>(*m_colorCloud2, *m_cloud2);
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr boundaryPoints1;
    pcl::PointCloud<pcl::PointXYZI>::Ptr boundaryPoints2;
    {
        m_boundaryExtractor->setInputCloud(m_cloud1);
        m_boundaryExtractor->setMatWidth(frame1.getDepthWidth());
        m_boundaryExtractor->setMatHeight(frame1.getDepthHeight());
        m_boundaryExtractor->setCx(frame1.getDevice()->cx());
        m_boundaryExtractor->setCy(frame1.getDevice()->cy());
        m_boundaryExtractor->setFx(frame1.getDevice()->fx());
        m_boundaryExtractor->setFy(frame1.getDevice()->fy());
        m_boundaryExtractor->setNormals(nullptr);
        m_boundaryExtractor->compute();
        boundaryPoints1 = m_boundaryExtractor->boundaryPoints();
        m_filteredCloud1 = m_boundaryExtractor->filteredCloud();

        m_boundaryExtractor->setInputCloud(m_cloud2);
        m_boundaryExtractor->setMatWidth(frame2.getDepthWidth());
        m_boundaryExtractor->setMatHeight(frame2.getDepthHeight());
        m_boundaryExtractor->setCx(frame2.getDevice()->cx());
        m_boundaryExtractor->setCy(frame2.getDevice()->cy());
        m_boundaryExtractor->setFx(frame2.getDevice()->fx());
        m_boundaryExtractor->setFy(frame2.getDevice()->fy());
        m_boundaryExtractor->setNormals(nullptr);
        m_boundaryExtractor->compute();
        boundaryPoints2 = m_boundaryExtractor->boundaryPoints();
        m_filteredCloud2 = m_boundaryExtractor->filteredCloud();
    }

    QList<LineSegment> lines1;
    QList<LineSegment> lines2;
    {
        lines1 = m_lineExtractor->compute(boundaryPoints1);
        m_mslCloud1 = m_lineExtractor->mslCloud();
        m_mslPointCloud1 = m_lineExtractor->mslPointCloud();
        m_diameter1 = m_lineExtractor->boundBoxDiameter();
        lines2 = m_lineExtractor->compute(boundaryPoints2);
        m_mslCloud2 = m_lineExtractor->mslCloud();
        m_mslPointCloud2 = m_lineExtractor->mslPointCloud();
        m_diameter2 = m_lineExtractor->boundBoxDiameter();
    }

    m_cloudViewer1->visualizer()->removeAllPointClouds();
    m_cloudViewer1->visualizer()->removeAllShapes();
    m_cloudViewer2->visualizer()->removeAllPointClouds();
    m_cloudViewer2->visualizer()->removeAllShapes();
    m_cloudViewer3->visualizer()->removeAllPointClouds();
    m_cloudViewer3->visualizer()->removeAllShapes();

    m_tree.reset(new pcl::KdTreeFLANN<DDBPLineExtractor::MSLPoint>());
    m_tree->setInputCloud(m_mslPointCloud2);

    // ��ʾ����
    {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> h1(m_colorCloud1, 127, 127, 127);
        m_cloudViewer2->visualizer()->addPointCloud(m_colorCloud1, h1, "cloud");
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> h2(m_colorCloud2, 127, 127, 127);
        m_cloudViewer3->visualizer()->addPointCloud(m_colorCloud2, h2, "cloud");
    }
    showCloudAndLines(m_cloudViewer2, lines1, m_mslCloud1);
    showCloudAndLines(m_cloudViewer3, lines2, m_mslCloud2);

    m_rotation = Eigen::Quaternionf::Identity();
    m_translation = Eigen::Vector3f::Zero();

    m_isInit = true;
}

void ToolWindowLineMatcher::compute()
{
    //m_rotation = m_lineMatcher->stepRotation(m_diameter1, m_mslPointCloud1, m_mslCloud1, m_diameter2, m_mslPointCloud2, m_mslCloud2, m_tree, m_rotationError, m_translationError);
}

void ToolWindowLineMatcher::showCloudAndLines(CloudViewer* viewer, QList<LineSegment>& lines, boost::shared_ptr<pcl::PointCloud<DDBPLineExtractor::MSL>>& mslCloud)
{
    {
        for (int i = 0; i < lines.size()/* && errors[i]*/; i++)
        {
            double r = rand() * 1.0 / RAND_MAX;
            double g = rand() * 1.0 / RAND_MAX;
            double b = rand() * 1.0 / RAND_MAX;
            LineSegment line = lines[i];
            std::string lineNo = "line_" + std::to_string(i);
            pcl::PointXYZI start, end;
            start.getVector3fMap() = line.start();
            end.getVector3fMap() = line.end();
            Eigen::Vector3f dir = line.direction();
            //m_cloudViewer->visualizer()->addArrow(end, start, r, g, b, 0, lineNo);
            viewer->visualizer()->addLine(start, end, r, g, b, lineNo);
            //viewer->visualizer()->addText3D(std::to_string(i), middle, 0.025, 1, 1, 1, textNo);
            viewer->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 4, lineNo);
        }
    }

    {
        for (int i = 0; i < mslCloud->size(); i++)
        {
            double r = rand() * 1.0 / RAND_MAX;
            double g = rand() * 1.0 / RAND_MAX;
            double b = rand() * 1.0 / RAND_MAX;
            DDBPLineExtractor::MSL msl = mslCloud->points[i];
            pcl::PointXYZI start, end, middle;
            start.getVector3fMap() = msl.getEndPoint(-3);
            end.getVector3fMap() = msl.getEndPoint(3);
            middle.getVector3fMap() = msl.point;
            QString lineName = QString("msl_%1").arg(i);
            std::string textNo = "text_" + std::to_string(i);
            viewer->visualizer()->addText3D(std::to_string(i), middle, 0.05, r, g, b, textNo);
            viewer->visualizer()->addLine(start, end, r, g, b, lineName.toStdString());
            viewer->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, lineName.toStdString());
        }
    }
}

void ToolWindowLineMatcher::showMatchedClouds()
{
    Eigen::Matrix4f rotMat(Eigen::Matrix4f::Identity());
    rotMat.topLeftCorner(3, 3) = m_rotation.toRotationMatrix();
    rotMat.topRightCorner(3, 1) = m_translation;
    pcl::transformPointCloud(*m_filteredCloud1, *m_filteredCloud1, rotMat);

    {
        m_cloudViewer1->visualizer()->removeAllPointClouds();
        m_cloudViewer1->visualizer()->removeAllShapes();

        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> h1(m_filteredCloud1, 255, 0, 0);
        m_cloudViewer1->visualizer()->addPointCloud(m_filteredCloud1, h1, "cloud1");
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> h2(m_filteredCloud2, 0, 0, 255);
        m_cloudViewer1->visualizer()->addPointCloud(m_filteredCloud2, h2, "cloud2");

        for (QMap<int, int>::iterator i = m_pairs.begin(); i != m_pairs.end(); i++)
        {
            {
                DDBPLineExtractor::MSL msl = m_mslCloud1->points[i.value()];
                pcl::PointXYZI start, end, middle;
                start.getVector3fMap() = msl.getEndPoint(-3);
                end.getVector3fMap() = msl.getEndPoint(3);
                middle.getVector3fMap() = msl.point;
                QString lineName = QString("msl_1_%1").arg(i.value());
                std::string textNo = "text_1_" + std::to_string(i.value());
                m_cloudViewer1->visualizer()->addText3D(std::to_string(i.value()), middle, 0.05, 255, 255, 0, textNo);
                m_cloudViewer1->visualizer()->addLine(start, end, 255, 255, 0, lineName.toStdString());
                m_cloudViewer1->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, lineName.toStdString());
            }
            {
                DDBPLineExtractor::MSL msl = m_mslCloud2->points[i.key()];
                pcl::PointXYZI start, end, middle;
                start.getVector3fMap() = msl.getEndPoint(-3);
                end.getVector3fMap() = msl.getEndPoint(3);
                middle.getVector3fMap() = msl.point;
                QString lineName = QString("msl_2_%1").arg(i.key());
                std::string textNo = "text_2_" + std::to_string(i.key());
                m_cloudViewer1->visualizer()->addText3D(std::to_string(i.key()), middle, 0.05, 0, 255, 255, textNo);
                m_cloudViewer1->visualizer()->addLine(start, end, 0, 255, 255, lineName.toStdString());
                m_cloudViewer1->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, lineName.toStdString());
            }
        }
    }
}

void ToolWindowLineMatcher::stepCompute()
{
}

void ToolWindowLineMatcher::onActionLoadDataSet()
{
    m_device.reset(new SensorReaderDevice);
    if (!m_device->open())
    {
        qDebug() << "Open device failed.";
        return;
    }

    m_ui->comboBoxFirstFrame->clear();
    m_ui->comboBoxSecondFrame->clear();
    for (int i = 0; i < m_device->totalFrames(); i++)
    {
        m_ui->comboBoxFirstFrame->addItem(QString::number(i));
        m_ui->comboBoxSecondFrame->addItem(QString::number(i));
    }
    m_ui->comboBoxFirstFrame->setCurrentIndex(0);
    m_ui->comboBoxSecondFrame->setCurrentIndex(1);
    m_isLoaded = true;
    updateWidgets();
}

void ToolWindowLineMatcher::onActionMatch()
{
    compute();
    updateWidgets();
}

void ToolWindowLineMatcher::onActionBeginStep()
{
    initCompute();

    m_isStepMode = true;
    updateWidgets();
}

void ToolWindowLineMatcher::onActionStepRotaionMatch()
{
    m_rotation = m_lineMatcher->stepRotation(m_diameter1, m_mslPointCloud1, m_mslCloud1, 
        m_diameter2, m_mslPointCloud2, m_mslCloud2, m_tree, m_rotationError, m_translationError, m_rotation, m_pairs);
    m_iteration++;

    showMatchedClouds();
    updateWidgets();
}

void ToolWindowLineMatcher::onActionStepTranslationMatch()
{
    m_translation = m_lineMatcher->stepTranslation(m_diameter1, m_mslPointCloud1, m_mslCloud1, 
        m_diameter2, m_mslPointCloud2, m_mslCloud2, m_tree, m_rotationError, m_translationError, m_rotation, m_translation, m_pairs);
    m_iteration++;

    showMatchedClouds();
    updateWidgets();
}

void ToolWindowLineMatcher::onActionReset()
{
    m_isInit = false;
    m_isStepMode = false;
    m_isLoaded = false;
    m_iteration = 0;

    updateWidgets();
}

void ToolWindowLineMatcher::updateWidgets()
{
    m_ui->actionLoad_Data_Set->setEnabled(!m_isLoaded);
    m_ui->actionMatch->setEnabled(!m_isInit && m_isLoaded);
    m_ui->actionBegin_Step->setEnabled(!m_isInit && m_isLoaded);
    m_ui->actionStep_Rotaion_Match->setEnabled(m_isInit && m_isStepMode);
    m_ui->actionStep_Translate_Match->setEnabled(m_isInit && m_isStepMode);
    m_ui->actionReset->setEnabled(m_isInit);

    m_ui->labelIteration->setText(QString::number(m_iteration));
    m_ui->labelRotationError->setText(QString::number(qRadiansToDegrees(m_rotationError)));
    m_ui->labelTranslationError->setText(QString::number(m_translationError));
}
