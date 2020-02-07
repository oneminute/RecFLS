#include "ToolWindowLineMatcher.h"
#include "ui_ToolWindowLineMatcher.h"

#include <QDebug>
#include <QtMath>

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
    connect(m_ui->actionStep_Rotaion_Match, &QAction::triggered, this, &ToolWindowLineMatcher::onActionStepRotaionMatch);
    connect(m_ui->actionStep_Translate_Match, &QAction::triggered, this, &ToolWindowLineMatcher::onActionStepTranslationMatch);
    connect(m_ui->actionReset, &QAction::triggered, this, &ToolWindowLineMatcher::onActionReset);

    connect(m_ui->comboBoxFirstFrame, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &ToolWindowLineMatcher::onComboBox1CurrentIndexChanged);

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
    {
        lines1 = m_lineExtractor->compute(boundaryPoints1);
        m_lineExtractor->extractLinesFromPlanes(m_planes1);
        m_mslCloud1 = m_lineExtractor->mslCloud();
        m_mslPointCloud1 = m_lineExtractor->mslPointCloud();
        m_diameter1 = m_lineExtractor->boundBoxDiameter();
        lines2 = m_lineExtractor->compute(boundaryPoints2);
        m_lineExtractor->extractLinesFromPlanes(m_planes2);
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

    m_tree.reset(new pcl::KdTreeFLANN<MSLPoint>());
    m_tree->setInputCloud(m_mslPointCloud2);
    qDebug() << "msl point cloud2:" << m_mslPointCloud2->size();

    // œ‘ æµ„‘∆
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

    m_rotationDelta = Eigen::Quaternionf::Identity();
    m_translationDelta = Eigen::Vector3f::Zero();

    m_isInit = true;
}

void ToolWindowLineMatcher::compute()
{
    initCompute();
    m_lineMatcher->compute(m_mslPointCloud1, m_mslCloud1, m_mslPointCloud2, m_mslCloud2);
    m_chains1 = m_lineMatcher->chains1();
    m_chains2 = m_lineMatcher->chains2();
    //m_rotation = m_lineMatcher->stepRotation(m_diameter1, m_mslPointCloud1, m_mslCloud1, m_diameter2, m_mslPointCloud2, m_mslCloud2, m_tree, m_rotationError, m_translationError);
    for (int i = 0; i < m_chains1.size(); i++)
    {
        LineMatcher::LineChain chain = m_chains1[i];
        QString ptName = QString("%1-%2").arg(chain.line1).arg(chain.line2);
        QString textName = ptName + "_txt";
        pcl::PointXYZ center;
        center.getVector3fMap() = (chain.point1 + chain.point2) / 2;

        m_cloudViewer2->visualizer()->addSphere(center, 0.025f, 255, 0, 0, ptName.toStdString());
        m_cloudViewer2->visualizer()->addText3D(ptName.toStdString(), center, 0.05, 1, 1, 1, textName.toStdString());
    }
    for (int i = 0; i < m_chains2.size(); i++)
    {
        LineMatcher::LineChain chain = m_chains2[i];
        QString ptName = QString("%1-%2").arg(chain.line1).arg(chain.line2);
        QString textName = ptName + "_txt";
        pcl::PointXYZ center;
        center.getVector3fMap() = (chain.point1 + chain.point2) / 2;

        m_cloudViewer3->visualizer()->addSphere(center, 0.025f, 255, 0, 0, ptName.toStdString());
        m_cloudViewer3->visualizer()->addText3D(ptName.toStdString(), center, 0.05, 1, 1, 1, textName.toStdString());
    }
    updateWidgets();
}

void ToolWindowLineMatcher::showCloudAndLines(CloudViewer* viewer, QList<Plane>& planes, QList<LineSegment>& lines, boost::shared_ptr<pcl::PointCloud<MSL>>& mslCloud)
{
    QColor color;
    for (int i = 0; i < planes.size(); i++)
    {
        QColor color;
        color.setHsv(255 * i / planes.size(), 255, 255);

        QString planeCloudName = QString("plane_cloud_%1").arg(i);
        QString planeName = QString("plane_%1").arg(i);
        QString sphereName = QString("sphere_%1").arg(i);
        QString normalName = QString("normal_%1").arg(i);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> iColor(planes[i].cloud, color.red(), color.green(), color.blue());
        viewer->visualizer()->addPointCloud(planes[i].cloud, iColor, planeCloudName.toStdString());
        viewer->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, planeCloudName.toStdString());

        pcl::PointXYZ center;
        pcl::PointXYZ end;
        center.getVector3fMap() = planes[i].point;
        end.getArray3fMap() = planes[i].point + planes[i].dir * 0.1f;
        qDebug() << "plane" << i << planes[i].parameters->values[0] << planes[i].parameters->values[1] << planes[i].parameters->values[2];
        viewer->visualizer()->addPlane(*planes[i].parameters, center.x, center.y, center.z, planeName.toStdString());
        //m_planeViewer->visualizer()->addSphere(center, 0.05f, 255, 0, 0, sphereName.toStdString());
        viewer->visualizer()->addLine(center, end, 255, 0, 0, normalName.toStdString());
    }

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
        viewer->visualizer()->addLine(start, end, r, g, b, lineNo);
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
        viewer->visualizer()->addText3D(std::to_string(i), middle, 0.05, r, g, b, textNo);
        //viewer->visualizer()->addLine(start, end, r, g, b, lineName.toStdString());
        viewer->visualizer()->addArrow(start, end, r, g, b, false, lineName.toStdString());
        viewer->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, lineName.toStdString());
    }
}

void ToolWindowLineMatcher::showMatchedClouds()
{
    Eigen::Matrix4f rotMat(Eigen::Matrix4f::Identity());
    rotMat.topLeftCorner(3, 3) = m_rotationDelta.toRotationMatrix();
    rotMat.topRightCorner(3, 1) = m_translationDelta;
    pcl::transformPointCloud(*m_colorCloud1, *m_colorCloud1, rotMat);

    {
        m_cloudViewer1->visualizer()->removeAllPointClouds();
        m_cloudViewer1->visualizer()->removeAllShapes();

        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> h1(m_colorCloud1);
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> h1(m_cloud1, 255, 0, 0);
        m_cloudViewer1->visualizer()->addPointCloud(m_colorCloud1, h1, "cloud1");
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> h2(m_colorCloud2);
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> h2(m_cloud2, 0, 0, 255);
        m_cloudViewer1->visualizer()->addPointCloud(m_colorCloud2, h2, "cloud2");

        for (QMap<int, int>::iterator i = m_pairs.begin(); i != m_pairs.end(); i++)
        {
            {
                MSL msl = m_mslCloud1->points[i.value()];
                pcl::PointXYZI start, end, middle;
                start.getVector3fMap() = msl.getEndPoint(-3);
                end.getVector3fMap() = msl.getEndPoint(3);
                middle.getVector3fMap() = msl.point;
                QString lineName = QString("msl_1_%1").arg(i.value());
                std::string textNo = "text_1_" + std::to_string(i.value());
                m_cloudViewer1->visualizer()->addText3D(std::to_string(i.value()), middle, 0.05, 255, 0, 0, textNo);
                m_cloudViewer1->visualizer()->addLine(start, end, 255, 255, 0, lineName.toStdString());
                m_cloudViewer1->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, lineName.toStdString());
            }
            {
                MSL msl = m_mslCloud2->points[i.key()];
                pcl::PointXYZI start, end, middle;
                start.getVector3fMap() = msl.getEndPoint(-3);
                end.getVector3fMap() = msl.getEndPoint(3);
                middle.getVector3fMap() = msl.point;
                QString lineName = QString("msl_2_%1").arg(i.key());
                std::string textNo = "text_2_" + std::to_string(i.key());
                m_cloudViewer1->visualizer()->addText3D(std::to_string(i.key()), middle, 0.05, 0, 0, 255, textNo);
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
