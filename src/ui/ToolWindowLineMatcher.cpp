#include "ToolWindowLineMatcher.h"
#include "ui_ToolWindowLineMatcher.h"

#include <QDebug>
#include <QtMath>
#include <QPushButton>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/pca.h>

#include "common/Parameters.h"
#include "util/Utils.h"

ToolWindowLineMatcher::ToolWindowLineMatcher(QWidget *parent) 
    : QMainWindow(parent)
    , m_ui(new Ui::ToolWindowLineMatcher)
    , m_isStepMode(false)
    , m_isInit(false)
    , m_isLoaded(false)
    , m_iteration(0)
    , m_diameter1(0)
    , m_diameter2(0)
    , m_rotationDelta(Eigen::Quaternionf::Identity())
    , m_translationDelta(Eigen::Vector3f::Zero())
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
    connect(m_ui->actionStep_Rotation_Match, &QAction::triggered, this, &ToolWindowLineMatcher::onActionStepRotationMatch);
    connect(m_ui->actionStep_Translate_Match, &QAction::triggered, this, &ToolWindowLineMatcher::onActionStepTranslationMatch);
    connect(m_ui->actionReset, &QAction::triggered, this, &ToolWindowLineMatcher::onActionReset);

    connect(m_ui->comboBoxFirstFrame, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &ToolWindowLineMatcher::onComboBox1CurrentIndexChanged);
    connect(m_ui->pushButtonShowLineChainPair, &QPushButton::clicked, this, &ToolWindowLineMatcher::onActionShowPair);

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
        m_lineExtractor.reset(new LineExtractor);
    }

    if (!m_lineMatcher)
    {
        m_lineMatcher.reset(new LineMatcher);
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
    m_boundaryExtractor->setPlaneDistanceThreshold(PARAMETERS.floatValue("plane_distance_threshold", 0.01f, "BoundaryExtractor"));

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

    m_ui->widgetFrame1->setImage(cvMat2QImage(frame1.colorMat()));
    m_ui->widgetFrame2->setImage(cvMat2QImage(frame2.colorMat()));

    pcl::IndicesPtr indices1(new std::vector<int>);
    pcl::IndicesPtr indices2(new std::vector<int>);
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
        m_planes1 = m_boundaryExtractor->planes();

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
        m_planes2 = m_boundaryExtractor->planes();
    }

    QList<LineSegment> lines1;
    QList<LineSegment> lines2;
    Eigen::Vector3f center1;
    Eigen::Vector3f center2;
    {
        lines1 = m_lineExtractor->compute(boundaryPoints1);
        m_lineExtractor->extractLinesFromPlanes(m_planes1);
        m_lineExtractor->generateLineChains();
        m_lineExtractor->generateDescriptors2();
        m_mslCloud1 = m_lineExtractor->mslCloud();
        m_mslPointCloud1 = m_lineExtractor->mslPointCloud();
        m_diameter1 = m_lineExtractor->boundBoxDiameter();
        m_chains1 = m_lineExtractor->chains();
        m_desc1 = m_lineExtractor->descriptors2();
        center1 = m_lineExtractor->lcLocalMiddle();

        lines2 = m_lineExtractor->compute(boundaryPoints2);
        m_lineExtractor->extractLinesFromPlanes(m_planes2);
        m_lineExtractor->generateLineChains();
        m_lineExtractor->generateDescriptors2();
        m_mslCloud2 = m_lineExtractor->mslCloud();
        m_mslPointCloud2 = m_lineExtractor->mslPointCloud();
        m_diameter2 = m_lineExtractor->boundBoxDiameter();
        m_chains2 = m_lineExtractor->chains();
        m_desc2 = m_lineExtractor->descriptors2();
        center2 = m_lineExtractor->lcLocalMiddle();
    }

    m_cloudViewer1->visualizer()->removeAllPointClouds();
    m_cloudViewer1->visualizer()->removeAllShapes();
    m_cloudViewer2->visualizer()->removeAllPointClouds();
    m_cloudViewer2->visualizer()->removeAllShapes();
    m_cloudViewer3->visualizer()->removeAllPointClouds();
    m_cloudViewer3->visualizer()->removeAllShapes();

    m_tree.reset(new pcl::KdTreeFLANN<MSLPoint>());
    m_tree->setInputCloud(m_mslPointCloud2);
    qDebug() << "msl point cloud2:" << m_mslPointCloud2->size();

    // 显示点云
    {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> h1(m_colorCloud1, 127, 127, 127);
        //m_cloudViewer2->visualizer()->addPointCloud(m_colorCloud1, h1, "cloud");
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> h2(m_colorCloud2, 127, 127, 127);
        //m_cloudViewer3->visualizer()->addPointCloud(m_colorCloud2, h2, "cloud");
    }
    showCloudAndLines(m_cloudViewer2, m_planes1, lines1, m_mslCloud1);
    showCloudAndLines(m_cloudViewer3, m_planes2, lines2, m_mslCloud2);
    //showCloudAndLines(m_cloudViewer2, QList<LineSegment>(), m_mslCloud1);
    //showCloudAndLines(m_cloudViewer3, QList<LineSegment>(), m_mslCloud2);

    //// 绘制帧局部原点
    //{
    //    pcl::PointXYZ center;
    //    center.getArray3fMap() = center1;
    //    m_cloudViewer2->visualizer()->addSphere(center, 0.1f, 255, 0, 0, "center");
    //    center.getArray3fMap() = center2;
    //    m_cloudViewer3->visualizer()->addSphere(center, 0.1f, 255, 0, 0, "center");
    //}

    m_rotationDelta = Eigen::Quaternionf::Identity();
    m_translationDelta = Eigen::Vector3f::Zero();

    m_isInit = true;
}

void ToolWindowLineMatcher::compute()
{
    initCompute();
    Eigen::Matrix4f M = m_lineMatcher->compute(m_chains1, m_mslCloud1, m_desc1, m_chains2, m_mslCloud2, m_desc2);

    m_rotation = M.topLeftCorner(3, 3);
    m_translation = M.topRightCorner(3, 1);

    m_pairs = m_lineMatcher->pairs();
    m_pairIndices = m_lineMatcher->pairIndices();

    m_ui->comboBoxLineChainPairs->clear();
    for (int i = 0; i < m_pairIndices.size(); i++)
    {
        int index2 = m_pairIndices[i];
        int index1 = m_pairs[index2];
        LineChain& lc1 = m_chains1[index1];
        LineChain& lc2 = m_chains2[index2];
        m_ui->comboBoxLineChainPairs->addItem(QString("%1[%3, %4] --> %2[%5, %6]").arg(index1).arg(index2).arg(lc1.lineNo1).arg(lc1.lineNo2).arg(lc2.lineNo1).arg(lc2.lineNo2));
    }
    
    showMatchedClouds();
    updateWidgets();
}

void ToolWindowLineMatcher::showCloudAndLines(CloudViewer* viewer, QList<Plane>& planes, QList<LineSegment>& lines, boost::shared_ptr<pcl::PointCloud<MSL>>& mslCloud)
{
    QColor color;
    //for (int i = 0; i < planes.size(); i++)
    //{
    //    QColor color;
    //    color.setHsv(255 * i / planes.size(), 255, 255);

    //    QString planeCloudName = QString("plane_cloud_%1").arg(i);
    //    QString planeName = QString("plane_%1").arg(i);
    //    QString sphereName = QString("sphere_%1").arg(i);
    //    QString normalName = QString("normal_%1").arg(i);
    //    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> iColor(planes[i].cloud, color.red(), color.green(), color.blue());
    //    viewer->visualizer()->addPointCloud(planes[i].cloud, iColor, planeCloudName.toStdString());
    //    viewer->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, planeCloudName.toStdString());

    //    pcl::PointXYZ center;
    //    pcl::PointXYZ end;
    //    center.getVector3fMap() = planes[i].point;
    //    end.getArray3fMap() = planes[i].point + planes[i].dir * 0.1f;
    //    qDebug() << "plane" << i << planes[i].parameters->values[0] << planes[i].parameters->values[1] << planes[i].parameters->values[2];
    //    viewer->visualizer()->addPlane(*planes[i].parameters, center.x, center.y, center.z, planeName.toStdString());
    //    //m_planeViewer->visualizer()->addSphere(center, 0.05f, 255, 0, 0, sphereName.toStdString());
    //    viewer->visualizer()->addLine(center, end, 255, 0, 0, normalName.toStdString());
    //}

    for (int i = 0; i < lines.size()/* && errors[i]*/; i++)
    {
        color.setHsv(i * 255 / lines.size(), 255, 255);
        double r = color.red();
        double g = color.green();
        double b = color.blue();
        LineSegment line = lines[i];
        std::string lineNo = "line_" + std::to_string(i);
        pcl::PointXYZI start, end;
        start.getVector3fMap() = line.start();
        end.getVector3fMap() = line.end();
        Eigen::Vector3f dir = line.direction();
        //m_cloudViewer->visualizer()->addArrow(end, start, r, g, b, 0, lineNo);
        viewer->visualizer()->addLine(start, end, 0, 255, 255, lineNo);
        //viewer->visualizer()->addText3D(std::to_string(i), middle, 0.025, 1, 1, 1, textNo);
        viewer->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 4, lineNo);
    }

    for (int i = 0; i < mslCloud->size(); i++)
    {
        color.setHsv(i * 255 / mslCloud->size(), 255, 255);
        double r = color.red();
        double g = color.green();
        double b = color.blue();
        MSL msl = mslCloud->points[i];
        pcl::PointXYZI start, end, middle;
        start.getVector3fMap() = msl.getEndPoint(-3);
        end.getVector3fMap() = msl.getEndPoint(3);
        middle.getVector3fMap() = msl.point;
        QString lineName = QString("msl_%1").arg(i);
        std::string textNo = "text_" + std::to_string(i);
        viewer->visualizer()->addText3D(std::to_string(i), middle, 0.05, 255, 255, 255, textNo);
        viewer->visualizer()->addLine(start, end, 0, 255, 255, lineName.toStdString());
        //viewer->visualizer()->addArrow(start, end, r, g, b, false, lineName.toStdString());
        viewer->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, lineName.toStdString());
    }
}

void ToolWindowLineMatcher::showMatchedClouds()
{
    Eigen::Matrix4f rotMat(Eigen::Matrix4f::Identity());
    rotMat.topLeftCorner(3, 3) = m_rotation;
    rotMat.topRightCorner(3, 1) = m_translation;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr m_tmpCloud1(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::transformPointCloud(*m_colorCloud1, *m_tmpCloud1, rotMat);

    {
        m_cloudViewer1->visualizer()->removeAllPointClouds();
        m_cloudViewer1->visualizer()->removeAllShapes();

        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> h1(m_tmpCloud1);
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> h1(m_tmpCloud1, 255, 0, 0);
        m_cloudViewer1->visualizer()->addPointCloud(m_tmpCloud1, h1, "cloud1");
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> h2(m_colorCloud2);
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> h2(m_colorCloud2, 0, 0, 255);
        m_cloudViewer1->visualizer()->addPointCloud(m_colorCloud2, h2, "cloud2");

        for (QMap<int, int>::iterator i = m_pairs.begin(); i != m_pairs.end(); i++)
        {
            LineChain lc1 = m_chains1[i.value()];
            LineChain lc2 = m_chains2[i.key()];
            {
                pcl::PointXYZI start, end, middle;

                //MSL msl = m_mslCloud1->points[lc1.line1];
                start.getVector3fMap() = lc1.line1.getEndPoint(-3);
                end.getVector3fMap() = lc1.line1.getEndPoint(3);
                middle.getVector3fMap() = lc1.line1.point;
                QString lineName = QString("msl_1_%1").arg(lc1.lineNo1);
                std::string textNo = "text_1_" + std::to_string(lc1.lineNo1);
                if (!m_cloudViewer1->visualizer()->contains(textNo))
                {
                    m_cloudViewer1->visualizer()->addText3D(std::to_string(lc1.lineNo1), middle, 0.05, 255, 0, 0, textNo);
                    m_cloudViewer1->visualizer()->addLine(start, end, 255, 255, 0, lineName.toStdString());
                    m_cloudViewer1->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, lineName.toStdString());
                }

                //msl = m_mslCloud1->points[lc1.line2];
                start.getVector3fMap() = lc1.line2.getEndPoint(-3);
                end.getVector3fMap() = lc1.line2.getEndPoint(3);
                middle.getVector3fMap() = lc1.line2.point;
                lineName = QString("msl_1_%1").arg(lc1.lineNo2);
                textNo = "text_1_" + std::to_string(lc1.lineNo2);
                if (!m_cloudViewer1->visualizer()->contains(textNo))
                {
                    m_cloudViewer1->visualizer()->addText3D(std::to_string(lc1.lineNo2), middle, 0.05, 255, 0, 0, textNo);
                    m_cloudViewer1->visualizer()->addLine(start, end, 255, 255, 0, lineName.toStdString());
                    m_cloudViewer1->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, lineName.toStdString());
                }
            }
            {
                pcl::PointXYZI start, end, middle;

                //MSL msl = m_mslCloud2->points[lc2.line1];
                start.getVector3fMap() = lc2.line1.getEndPoint(-3);
                end.getVector3fMap() = lc2.line1.getEndPoint(3);
                middle.getVector3fMap() = lc2.line1.point;
                QString lineName = QString("msl_2_%1").arg(lc2.lineNo1);
                std::string textNo = "text_2_" + std::to_string(lc2.lineNo1);
                if (!m_cloudViewer1->visualizer()->contains(textNo))
                {
                    m_cloudViewer1->visualizer()->addText3D(std::to_string(lc2.lineNo1), middle, 0.05, 0, 0, 255, textNo);
                    m_cloudViewer1->visualizer()->addLine(start, end, 0, 255, 255, lineName.toStdString());
                    m_cloudViewer1->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, lineName.toStdString());
                }

                //msl = m_mslCloud2->points[lc2.line2];
                start.getVector3fMap() = lc2.line2.getEndPoint(-3);
                end.getVector3fMap() = lc2.line2.getEndPoint(3);
                middle.getVector3fMap() = lc2.line2.point;
                lineName = QString("msl_2_%1").arg(lc2.lineNo2);
                textNo = "text_2_" + std::to_string(lc2.lineNo2);
                if (!m_cloudViewer1->visualizer()->contains(textNo))
                {
                    m_cloudViewer1->visualizer()->addText3D(std::to_string(lc2.lineNo2), middle, 0.05, 0, 0, 255, textNo);
                    m_cloudViewer1->visualizer()->addLine(start, end, 0, 255, 255, lineName.toStdString());
                    m_cloudViewer1->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, lineName.toStdString());
                }
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

    int index1 = m_ui->comboBoxFirstFrame->currentIndex();
    int index2 = m_ui->comboBoxSecondFrame->currentIndex();
    if (index1 < 0)
    {
        index1 = 0;
        index2 = 1;
    }
    m_ui->comboBoxFirstFrame->clear();
    m_ui->comboBoxSecondFrame->clear();
    for (int i = 0; i < m_device->totalFrames(); i++)
    {
        m_ui->comboBoxFirstFrame->addItem(QString::number(i));
        m_ui->comboBoxSecondFrame->addItem(QString::number(i));
    }
    m_ui->comboBoxFirstFrame->setCurrentIndex(index1);
    m_ui->comboBoxSecondFrame->setCurrentIndex(index2);
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

void ToolWindowLineMatcher::onActionStepRotationMatch()
{
    m_rotationDelta = m_lineMatcher->stepRotation(m_diameter1, m_mslPointCloud1, m_mslCloud1, 
        m_diameter2, m_mslPointCloud2, m_mslCloud2, m_tree, m_rotationError, m_translationError, m_pairs);
    m_translationDelta = Eigen::Vector3f::Zero();
    m_iteration++;

    showMatchedClouds();
    updateWidgets();
}

void ToolWindowLineMatcher::onActionStepTranslationMatch()
{
    m_translationDelta = m_lineMatcher->stepTranslation(m_mslCloud1, m_mslCloud2, m_tree, m_translationError, m_pairs);
    m_rotationDelta = Eigen::Quaternionf::Identity();
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

void ToolWindowLineMatcher::onComboBox1CurrentIndexChanged(int index)
{
    if (index != m_ui->comboBoxSecondFrame->count() - 1)
        m_ui->comboBoxSecondFrame->setCurrentIndex(index + 1);
}

void ToolWindowLineMatcher::onActionShowPair(bool isChecked)
{
    qDebug() << "onActionShowPair";
    m_cloudViewer2->visualizer()->removeShape("chain_line_1");
    m_cloudViewer2->visualizer()->removeShape("chain_line_2");
    m_cloudViewer2->visualizer()->removeShape("chain_line");
    m_cloudViewer2->visualizer()->removeShape("xaxis");
    m_cloudViewer2->visualizer()->removeShape("yaxis");
    m_cloudViewer2->visualizer()->removeShape("zaxis");

    m_cloudViewer3->visualizer()->removeShape("chain_line_1");
    m_cloudViewer3->visualizer()->removeShape("chain_line_2");
    m_cloudViewer3->visualizer()->removeShape("chain_line");
    m_cloudViewer3->visualizer()->removeShape("xaxis");
    m_cloudViewer3->visualizer()->removeShape("yaxis");
    m_cloudViewer3->visualizer()->removeShape("zaxis");

    int index2 = m_pairIndices[m_ui->comboBoxLineChainPairs->currentIndex()];
    int index1 = m_pairs[index2];
    LineChain lc1 = m_chains1[index1];
    LineChain lc2 = m_chains2[index2];
    qDebug() << index1 << lc1.name() << index2 << lc2.name();

    {
        pcl::PointXYZ start, end;
        //MSL msl = m_mslCloud1->points[lc1.line1];
        start.getVector3fMap() = lc1.line1.getEndPoint(-3);
        end.getVector3fMap() = lc1.line1.getEndPoint(3);
        m_cloudViewer2->visualizer()->addLine(start, end, 255, 0, 0, "chain_line_1");
        m_cloudViewer2->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "chain_line_1");

        //msl = m_mslCloud1->points[lc1.line2];
        start.getVector3fMap() = lc1.line2.getEndPoint(-3);
        end.getVector3fMap() = lc1.line2.getEndPoint(3);
        m_cloudViewer2->visualizer()->addLine(start, end, 0, 0, 255, "chain_line_2");
        m_cloudViewer2->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "chain_line_2");

        start.getVector3fMap() = lc1.point1;
        end.getVector3fMap() = lc1.point2;
        m_cloudViewer2->visualizer()->addLine(start, end, 0, 255, 0, "chain_line");
        m_cloudViewer2->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "chain_line");

        end.getVector3fMap() = lc1.point;
        start.getVector3fMap() = lc1.point + lc1.xLocal * 0.2f;
        m_cloudViewer2->visualizer()->addArrow(start, end, 255, 0, 0, false, "xaxis");
        start.getVector3fMap() = lc1.point + lc1.yLocal * 0.2f;
        m_cloudViewer2->visualizer()->addArrow(start, end, 0, 255, 0, false, "yaxis");
        start.getVector3fMap() = lc1.point + lc1.zLocal * 0.2f;
        m_cloudViewer2->visualizer()->addArrow(start, end, 0, 0, 255, false, "zaxis");
    }

    {
        pcl::PointXYZ start, end;
        //MSL msl = m_mslCloud2->points[lc2.line1];
        start.getVector3fMap() = lc2.line1.getEndPoint(-3);
        end.getVector3fMap() = lc2.line1.getEndPoint(3);
        m_cloudViewer3->visualizer()->addLine(start, end, 255, 0, 0, "chain_line_1");
        m_cloudViewer3->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "chain_line_1");

        //msl = m_mslCloud2->points[lc2.line2];
        start.getVector3fMap() = lc2.line2.getEndPoint(-3);
        end.getVector3fMap() = lc2.line2.getEndPoint(3);
        m_cloudViewer3->visualizer()->addLine(start, end, 0, 0, 255, "chain_line_2");
        m_cloudViewer3->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "chain_line_2");

        start.getVector3fMap() = lc2.point1;
        end.getVector3fMap() = lc2.point2;
        m_cloudViewer3->visualizer()->addLine(start, end, 0, 255, 0, "chain_line");
        m_cloudViewer3->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "chain_line");

        end.getVector3fMap() = lc2.point;
        start.getVector3fMap() = lc2.point + lc2.xLocal * 0.2f;
        m_cloudViewer3->visualizer()->addArrow(start, end, 255, 0, 0, false, "xaxis");
        start.getVector3fMap() = lc2.point + lc2.yLocal * 0.2f;
        m_cloudViewer3->visualizer()->addArrow(start, end, 0, 255, 0, false, "yaxis");
        start.getVector3fMap() = lc2.point + lc2.zLocal * 0.2f;
        m_cloudViewer3->visualizer()->addArrow(start, end, 0, 0, 255, false, "zaxis");
    }
}

void ToolWindowLineMatcher::updateWidgets()
{
    m_ui->actionLoad_Data_Set->setEnabled(!m_isLoaded);
    m_ui->actionMatch->setEnabled(!m_isInit && m_isLoaded);
    m_ui->actionBegin_Step->setEnabled(!m_isInit && m_isLoaded);
    m_ui->actionStep_Rotation_Match->setEnabled(m_isInit && m_isStepMode);
    m_ui->actionStep_Translate_Match->setEnabled(m_isInit && m_isStepMode);
    m_ui->actionReset->setEnabled(m_isInit);

    m_ui->labelIteration->setText(QString::number(m_iteration));
    m_ui->labelRotationError->setText(QString::number(qRadiansToDegrees(m_rotationError)));
    m_ui->labelTranslationError->setText(QString::number(m_translationError));
}
