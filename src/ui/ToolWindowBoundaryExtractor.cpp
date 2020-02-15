#include "ToolWindowBoundaryExtractor.h"
#include "ui_ToolWindowBoundaryExtractor.h"

#include <QDebug>
#include <QAction>
#include <QFileDialog>
#include <QDir>
#include <QtMath>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>

#include <opencv2/opencv.hpp>

#include "util/Utils.h"
#include "util/StopWatch.h"
#include "common/Parameters.h"
#include "common/Frame.h"

ToolWindowBoundaryExtractor::ToolWindowBoundaryExtractor(QWidget *parent) 
    : QMainWindow(parent)
    , m_ui(new Ui::ToolWindowBoundaryExtractor)
{
    m_ui->setupUi(this);

    m_cloudViewer = new CloudViewer;
    m_cloudViewer->setCameraPosition(0, 0, -1.5f, 0, 0, 0, 0, -1, 0);
    m_projectedCloudViewer = new CloudViewer;
    m_projectedCloudViewer->setCameraPosition(0, 0, -1.5f, 0, 0, 0, 0, 1, 0);
    m_planeViewer = new CloudViewer;
    m_planeViewer->setCameraPosition(0, 0, -1.5f, 0, 0, 0, 0, -1, 0);
    m_depthViewer = new ImageViewer;
    m_depthViewer2 = new ImageViewer;

    m_ui->verticalLayout1->addWidget(m_cloudViewer);
    m_ui->verticalLayout1->addWidget(m_planeViewer);
    m_ui->verticalLayout2->addWidget(m_projectedCloudViewer);
    m_ui->verticalLayout2->addWidget(m_depthViewer);
    m_ui->verticalLayout2->addWidget(m_depthViewer2);

    //m_ui->comboBoxDownsamplingMethod->setCurrentIndex(PARAMETERS.intValue();

    connect(m_ui->actionLoad_Data_Set, &QAction::triggered, this, &ToolWindowBoundaryExtractor::onActionLoadDataSet);
    connect(m_ui->actionCompute, &QAction::triggered, this, &ToolWindowBoundaryExtractor::onActionCompute);
    connect(m_ui->actionSave, &QAction::triggered, this, &ToolWindowBoundaryExtractor::onActionSave);
    connect(m_ui->actionSave_Config, &QAction::triggered, this, &ToolWindowBoundaryExtractor::onActionSaveConfig);

    m_ui->comboBoxDownsamplingMethod->setCurrentIndex(PARAMETERS.intValue("downsampling_method", 0, "BoundaryExtractor"));
    m_ui->checkBoxEnableRemovalFilter->setChecked(PARAMETERS.boolValue("enable_removal_filter", false, "BoundaryExtractor"));
    m_ui->doubleSpinBoxDownsampleLeafSize->setValue(PARAMETERS.floatValue("downsample_leaf_size", 0.0075f, "BoundaryExtractor"));
    m_ui->doubleSpinBoxOutlierRemovalMeanK->setValue(PARAMETERS.floatValue("outlier_removal_mean_k", 20.f, "BoundaryExtractor"));
    m_ui->doubleSpinBoxStddevMulThresh->setValue(PARAMETERS.floatValue("std_dev_mul_thresh", 1.f, "BoundaryExtractor"));
    m_ui->doubleSpinBoxGaussianSigma->setValue(PARAMETERS.floatValue("gaussian_sigma", 4.f, "BoundaryExtractor"));
    m_ui->doubleSpinBoxGaussianRSigma->setValue(PARAMETERS.floatValue("gaussian_r_sigma", 4.f, "BoundaryExtractor"));
    m_ui->doubleSpinBoxGaussianRadiusSearch->setValue(PARAMETERS.floatValue("gaussian_radius_search", 0.05f, "BoundaryExtractor"));
    m_ui->doubleSpinBoxNormalsRadiusSearch->setValue(PARAMETERS.floatValue("normals_radius_search", 0.05f, "BoundaryExtractor"));
    m_ui->doubleSpinBoxRadiusSearch->setValue(PARAMETERS.floatValue("boundary_radius_search", 0.1f, "BoundaryExtractor"));
    m_ui->spinBoxAngleThresholdDivision->setValue(qRound(M_PI / PARAMETERS.floatValue("boundary_angle_threshold", M_PI_2, "BoundaryExtractor")));
    m_ui->spinBoxBorderLeft->setValue(PARAMETERS.floatValue("border_left", 26, "BoundaryExtractor"));
    m_ui->spinBoxBorderRight->setValue(PARAMETERS.floatValue("border_right", 22, "BoundaryExtractor"));
    m_ui->spinBoxBorderTop->setValue(PARAMETERS.floatValue("border_top", 16, "BoundaryExtractor"));
    m_ui->spinBoxBorderBottom->setValue(PARAMETERS.floatValue("border_bottom", 16, "BoundaryExtractor"));
    m_ui->doubleSpinBoxProjectedRadiusSearch->setValue(qDegreesToRadians(PARAMETERS.floatValue("projected_radius_search", 5, "BoundaryExtractor")));
    m_ui->doubleSpinBoxVeilDistanceThreshold->setValue(PARAMETERS.floatValue("veil_distance_threshold", 0.1f, "BoundaryExtractor"));
    m_ui->doubleSpinBoxPlaneDistanceThreshold->setValue(PARAMETERS.floatValue("plane_distance_threshold", 0.01f, "BoundaryExtractor"));
}

ToolWindowBoundaryExtractor::~ToolWindowBoundaryExtractor()
{
}

void ToolWindowBoundaryExtractor::onActionCompute()
{
    m_boundaryExtractor.reset(new BoundaryExtractor);
    m_boundaryExtractor->setDownsamplingMethod(m_ui->comboBoxDownsamplingMethod->currentIndex());
    m_boundaryExtractor->setEnableRemovalFilter(m_ui->checkBoxEnableRemovalFilter->isChecked());
    m_boundaryExtractor->setDownsampleLeafSize(m_ui->doubleSpinBoxDownsampleLeafSize->value());
    m_boundaryExtractor->setOutlierRemovalMeanK(m_ui->doubleSpinBoxOutlierRemovalMeanK->value());
    m_boundaryExtractor->setStddevMulThresh(m_ui->doubleSpinBoxStddevMulThresh->value());
    m_boundaryExtractor->setGaussianSigma(m_ui->doubleSpinBoxGaussianSigma->value());
    m_boundaryExtractor->setGaussianRSigma(m_ui->doubleSpinBoxGaussianRSigma->value());
    m_boundaryExtractor->setGaussianRadiusSearch(m_ui->doubleSpinBoxGaussianRadiusSearch->value());
    m_boundaryExtractor->setNormalsRadiusSearch(m_ui->doubleSpinBoxNormalsRadiusSearch->value());
    m_boundaryExtractor->setBoundaryRadiusSearch(m_ui->doubleSpinBoxRadiusSearch->value());
    m_boundaryExtractor->setBoundaryAngleThreshold(M_PI / m_ui->spinBoxAngleThresholdDivision->value());
    m_boundaryExtractor->setBorderLeft(m_ui->spinBoxBorderLeft->value());
    m_boundaryExtractor->setBorderRight(m_ui->spinBoxBorderRight->value());
    m_boundaryExtractor->setBorderTop(m_ui->spinBoxBorderTop->value());
    m_boundaryExtractor->setBorderBottom(m_ui->spinBoxBorderBottom->value());
    m_boundaryExtractor->setProjectedRadiusSearch(qDegreesToRadians(m_ui->doubleSpinBoxProjectedRadiusSearch->value()));
    m_boundaryExtractor->setVeilDistanceThreshold(m_ui->doubleSpinBoxVeilDistanceThreshold->value());
    m_boundaryExtractor->setCrossPointsRadiusSearch(m_ui->doubleSpinBoxCrossPointsRadiusSearch->value());
    m_boundaryExtractor->setCrossPointsClusterTolerance(m_ui->doubleSpinBoxCrossPointsClusterTolerance->value());
    m_boundaryExtractor->setCurvatureThreshold(m_ui->doubleSpinBoxCurvatureThreshold->value());
    m_boundaryExtractor->setMinNormalClusters(m_ui->spinBoxMinNormalClusters->value());
    m_boundaryExtractor->setMaxNormalCulsters(m_ui->spinBoxMaxNormalClusters->value());
    m_boundaryExtractor->setPlaneDistanceThreshold(m_ui->doubleSpinBoxPlaneDistanceThreshold->value());

    m_lineExtractor.reset(new LineExtractor);
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

    m_cloudViewer->visualizer()->removeAllPointClouds();
    m_cloudViewer->visualizer()->removeAllShapes();
    m_projectedCloudViewer->visualizer()->removeAllPointClouds();
    m_projectedCloudViewer->visualizer()->removeAllShapes();
    m_planeViewer->visualizer()->removeAllPointClouds();
    m_planeViewer->visualizer()->removeAllShapes();

    int frameIndex = m_ui->comboBoxFrameIndex->currentIndex();
    Frame frame = m_device->getFrame(frameIndex);
    pcl::IndicesPtr indices(new std::vector<int>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorCloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    colorCloud = frame.getCloud(*indices);
    pcl::copyPointCloud<pcl::PointXYZRGB, pcl::PointXYZ>(*colorCloud, *cloud);

    m_boundaryExtractor->setInputCloud(cloud);
    m_boundaryExtractor->setMatWidth(frame.getDepthWidth());
    m_boundaryExtractor->setMatHeight(frame.getDepthHeight());
    m_boundaryExtractor->setCx(frame.getDevice()->cx());
    m_boundaryExtractor->setCy(frame.getDevice()->cy());
    m_boundaryExtractor->setFx(frame.getDevice()->fx());
    m_boundaryExtractor->setFy(frame.getDevice()->fy());
    //m_boundaryExtractor->setIndices(indices);
    m_allBoundary = m_boundaryExtractor->compute();
    pcl::PointCloud<pcl::PointXYZ>::Ptr filteredCloud = m_boundaryExtractor->filteredCloud();
    pcl::PointCloud<pcl::Normal>::Ptr normalsCloud = m_boundaryExtractor->normals();

    pcl::PointCloud<pcl::PointXYZI>::Ptr m_projectedCloud = m_boundaryExtractor->projectedCloud();
    m_boundaryPoints = m_boundaryExtractor->boundaryPoints();
    cv::Mat boundaryMat = m_boundaryExtractor->boundaryMat();
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_veilPoints = m_boundaryExtractor->veilPoints();
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_borderPoints = m_boundaryExtractor->borderPoints();

    m_depthViewer->setImage(cvMat2QImage(frame.depthMat(), false));
    m_depthViewer2->setImage(cvMat2QImage(boundaryMat, false));
    if (m_ui->radioButtonShowColor->isChecked())
    {
        m_cloudViewer->addCloud("scene cloud", colorCloud);
    }
    else if (m_ui->radioButtonShowNoColor->isChecked())
    {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> yColor(filteredCloud, 255, 255, 127);
        m_cloudViewer->visualizer()->addPointCloud(filteredCloud, yColor, "scene cloud");
    }

    if (m_ui->checkBoxShowNormals->isChecked())
    {
        m_cloudViewer->visualizer()->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(filteredCloud, normalsCloud, 10, 0.02f);
    }

    if (m_projectedCloud) {
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> iColor(m_projectedCloud, "intensity");
        m_projectedCloudViewer->visualizer()->addPointCloud(m_projectedCloud, iColor, "projected cloud");
        m_projectedCloudViewer->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "projected cloud");
    }

    if (m_boundaryPoints) {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> rColor(m_boundaryPoints, 255, 0, 0);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> gColor(m_veilPoints, 0, 255, 0);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> bColor(m_borderPoints, 0, 0, 255);

        m_cloudViewer->visualizer()->addPointCloud(m_boundaryPoints, rColor, "boundary points");
        m_cloudViewer->visualizer()->addPointCloud(m_veilPoints, gColor, "veil points");
        m_cloudViewer->visualizer()->addPointCloud(m_borderPoints, bColor, "border points");
        m_cloudViewer->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "boundary points");
        m_cloudViewer->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "veil points");
        m_cloudViewer->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "border points");
    }

    //{
    //    QList<Plane> planes = m_boundaryExtractor->planes();
    //    m_lineExtractor->extractLinesFromPlanes(planes);
    //    pcl::PointCloud<MSL>::Ptr mslCloud = m_lineExtractor->mslCloud();

    //    for (int i = 0; i < planes.size(); i++)
    //    {
    //        QColor color;
    //        color.setHsv(255 * i / planes.size(), 255, 255);

    //        QString planeCloudName = QString("plane_cloud_%1").arg(i);
    //        QString planeName = QString("plane_%1").arg(i);
    //        QString sphereName = QString("sphere_%1").arg(i);
    //        QString normalName = QString("normal_%1").arg(i);
    //        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> iColor(planes[i].cloud, color.red(), color.green(), color.blue());
    //        m_planeViewer->visualizer()->addPointCloud(planes[i].cloud, iColor, planeCloudName.toStdString());
    //        m_planeViewer->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, planeCloudName.toStdString());

    //        pcl::PointXYZ center;
    //        pcl::PointXYZ end;
    //        center.getVector3fMap() = planes[i].point;
    //        end.getArray3fMap() = planes[i].point + planes[i].dir * 0.1f;
    //        qDebug() << "plane" << i << planes[i].parameters->values[0] << planes[i].parameters->values[1] << planes[i].parameters->values[2];
    //        m_planeViewer->visualizer()->addPlane(*planes[i].parameters, center.x, center.y, center.z, planeName.toStdString());
    //        //m_planeViewer->visualizer()->addSphere(center, 0.05f, 255, 0, 0, sphereName.toStdString());
    //        m_planeViewer->visualizer()->addLine(center, end, 255, 0, 0, normalName.toStdString());
    //    }

    //    for (int i = 0; i < mslCloud->size(); i++)
    //    {
    //        QColor color;
    //        color.setHsv(i * 255 / mslCloud->size(), 255, 255);
    //        double r = color.red();
    //        double g = color.green();
    //        double b = color.blue();
    //        MSL msl = mslCloud->points[i];
    //        pcl::PointXYZI start, end, middle;
    //        start.getVector3fMap() = msl.getEndPoint(-3);
    //        end.getVector3fMap() = msl.getEndPoint(3);
    //        middle.getVector3fMap() = msl.point;
    //        QString lineName = QString("msl_%1").arg(i);
    //        QString sphereName = QString("msl_sphere_%1").arg(i);
    //        std::string textNo = "text_" + std::to_string(i);
    //        m_planeViewer->visualizer()->addText3D(std::to_string(i), middle, 0.05, r, g, b, textNo);
    //        m_planeViewer->visualizer()->addLine(start, end, r, g, b, lineName.toStdString());
    //        m_planeViewer->visualizer()->addSphere(middle, 0.05f, r, g, b, sphereName.toStdString());
    //        m_planeViewer->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, lineName.toStdString());
    //    }
    //}

    StopWatch::instance().debugPrint();
}

void ToolWindowBoundaryExtractor::onActionSave()
{
    QString fileName = QString("%1/%2_%3_boundary.pcd").arg(QDir::currentPath())
        .arg(Parameters::Global().stringValue("sample_path", "samples/office3.sens", "Device_SensorReader")).arg(m_ui->comboBoxFrameIndex->currentIndex());
    fileName = QFileDialog::getSaveFileName(this, tr("Save Boundaries"), QDir::currentPath(), tr("Polygon Files (*.obj *.ply *.pcd)"));
    qDebug() << "saving file" << fileName;
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::copyPointCloud(*m_boundaryPoints, cloud);
    cloud.width = cloud.size();
    cloud.height = 1;
    cloud.is_dense = true;
    pcl::io::savePCDFile<pcl::PointXYZ>(fileName.toStdString(), cloud);
}

void ToolWindowBoundaryExtractor::onActionSaveConfig()
{
    PARAMETERS.setValue("downsampling_method", m_ui->comboBoxDownsamplingMethod->currentIndex(), "BoundaryExtractor");
    PARAMETERS.setValue("enable_removal_filter", m_ui->checkBoxEnableRemovalFilter->isChecked(), "BoundaryExtractor");
    PARAMETERS.setValue("downsample_leaf_size", m_ui->doubleSpinBoxDownsampleLeafSize->value(), "BoundaryExtractor");
    PARAMETERS.setValue("outlier_removal_mean_k", m_ui->doubleSpinBoxOutlierRemovalMeanK->value(), "BoundaryExtractor");
    PARAMETERS.setValue("std_dev_mul_thresh", m_ui->doubleSpinBoxStddevMulThresh->value(), "BoundaryExtractor");
    PARAMETERS.setValue("gaussian_sigma", m_ui->doubleSpinBoxGaussianSigma->value(), "BoundaryExtractor");
    PARAMETERS.setValue("gaussian_r_sigma", m_ui->doubleSpinBoxGaussianRSigma->value(), "BoundaryExtractor");
    PARAMETERS.setValue("gaussian_radius_search", m_ui->doubleSpinBoxGaussianRadiusSearch->value(), "BoundaryExtractor");
    PARAMETERS.setValue("normals_radius_search", m_ui->doubleSpinBoxNormalsRadiusSearch->value(), "BoundaryExtractor");
    PARAMETERS.setValue("boundary_radius_search", m_ui->doubleSpinBoxRadiusSearch->value(), "BoundaryExtractor");
    PARAMETERS.setValue("boundary_angle_threshold", M_PI / m_ui->spinBoxAngleThresholdDivision->value(), "BoundaryExtractor");
    PARAMETERS.setValue("border_left", m_ui->spinBoxBorderLeft->value(), "BoundaryExtractor");
    PARAMETERS.setValue("border_right", m_ui->spinBoxBorderRight->value(), "BoundaryExtractor");
    PARAMETERS.setValue("border_top", m_ui->spinBoxBorderTop->value(), "BoundaryExtractor");
    PARAMETERS.setValue("border_bottom", m_ui->spinBoxBorderBottom->value(), "BoundaryExtractor");
    PARAMETERS.setValue("projected_radius_search", m_ui->doubleSpinBoxProjectedRadiusSearch->value(), "BoundaryExtractor");
    PARAMETERS.setValue("veil_distance_threshold", m_ui->doubleSpinBoxVeilDistanceThreshold->value(), "BoundaryExtractor");
    PARAMETERS.setValue("cross_points_radius_search", m_ui->doubleSpinBoxCrossPointsRadiusSearch->value(), "BoundaryExtractor");
    PARAMETERS.setValue("cross_points_cluster_tolerance", m_ui->doubleSpinBoxCrossPointsClusterTolerance->value(), "BoundaryExtractor");
    PARAMETERS.setValue("curvature_threshold", m_ui->doubleSpinBoxCurvatureThreshold->value(), "BoundaryExtractor");
    PARAMETERS.setValue("min_normal_clusters", m_ui->spinBoxMinNormalClusters->value(), "BoundaryExtractor");
    PARAMETERS.setValue("max_normal_clusters", m_ui->spinBoxMaxNormalClusters->value(), "BoundaryExtractor");
    PARAMETERS.setValue("plane_distance_threshold", m_ui->doubleSpinBoxPlaneDistanceThreshold->value(), "BoundaryExtractor");

    PARAMETERS.save();
}

void ToolWindowBoundaryExtractor::onActionLoadDataSet()
{
    m_device.reset(new SensorReaderDevice);
    if (!m_device->open())
    {
        qDebug() << "Open device failed.";
        return;
    }

    m_ui->comboBoxFrameIndex->clear();
    for (int i = 0; i < m_device->totalFrames(); i++)
    {
        m_ui->comboBoxFrameIndex->addItem(QString::number(i));
    }
}
