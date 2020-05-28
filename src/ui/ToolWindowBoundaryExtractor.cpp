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
    , m_init(false)
{
    m_ui->setupUi(this);

    m_cloudViewer = new CloudViewer;
    m_cloudViewer->setCameraPosition(0, 0, -1.5f, 0, 0, 0, 0, -1, 0);
    //m_projectedCloudViewer = new CloudViewer;
    //m_projectedCloudViewer->setCameraPosition(0, 0, -1.5f, 0, 0, 0, 0, 1, 0);
    //m_planeViewer = new CloudViewer;
    //m_planeViewer->setCameraPosition(0, 0, -1.5f, 0, 0, 0, 0, -1, 0);
    //m_depthViewer = new ImageViewer;
    m_depthViewer2 = new ImageViewer;

    m_ui->verticalLayout1->addWidget(m_cloudViewer);
    //m_ui->verticalLayout1->addWidget(m_planeViewer);
    //m_ui->verticalLayout2->addWidget(m_projectedCloudViewer);
    //m_ui->verticalLayout2->addWidget(m_depthViewer);
    m_ui->verticalLayout2->addWidget(m_depthViewer2);

    //m_ui->comboBoxDownsamplingMethod->setCurrentIndex(PARAMETERS.intValue();

    connect(m_ui->actionLoad_Data_Set, &QAction::triggered, this, &ToolWindowBoundaryExtractor::onActionLoadDataSet);
    //connect(m_ui->actionCompute, &QAction::triggered, this, &ToolWindowBoundaryExtractor::onActionCompute);
    connect(m_ui->actionCompute_GPU, &QAction::triggered, this, &ToolWindowBoundaryExtractor::onActionComputeGPU);
    connect(m_ui->actionCompute_VBRG, &QAction::triggered, this, &ToolWindowBoundaryExtractor::onActionComputeVBRG);
    connect(m_ui->actionSave, &QAction::triggered, this, &ToolWindowBoundaryExtractor::onActionSave);
    connect(m_ui->actionSave_Config, &QAction::triggered, this, &ToolWindowBoundaryExtractor::onActionSaveConfig);
}

ToolWindowBoundaryExtractor::~ToolWindowBoundaryExtractor()
{
}

void ToolWindowBoundaryExtractor::init()
{
    m_boundaryExtractor.reset(new BoundaryExtractor);
    m_boundaryExtractor->setBorderLeft(Settings::BoundaryExtractor_BorderLeft.value());
    m_boundaryExtractor->setBorderRight(Settings::BoundaryExtractor_BorderRight.value());
    m_boundaryExtractor->setBorderTop(Settings::BoundaryExtractor_BorderTop.value());
    m_boundaryExtractor->setBorderBottom(Settings::BoundaryExtractor_BorderBottom.value());
    m_boundaryExtractor->setMinDepth(Settings::BoundaryExtractor_MinDepth.value());
    m_boundaryExtractor->setMaxDepth(Settings::BoundaryExtractor_MaxDepth.value());
    m_boundaryExtractor->setCudaNormalKernalRadius(Settings::BoundaryExtractor_CudaNormalKernalRadius.intValue());
    m_boundaryExtractor->setCudaNormalKnnRadius(Settings::BoundaryExtractor_CudaNormalKnnRadius.value());
    m_boundaryExtractor->setCudaBEDistance(Settings::BoundaryExtractor_CudaBEDistance.value());
    m_boundaryExtractor->setCudaBEAngleThreshold(Settings::BoundaryExtractor_CudaBEAngleThreshold.value());
    m_boundaryExtractor->setCudaBEKernalRadius(Settings::BoundaryExtractor_CudaBEKernalRadius.value());
    m_boundaryExtractor->setCudaGaussianSigma(Settings::BoundaryExtractor_CudaGaussianSigma.value());
    m_boundaryExtractor->setCudaGaussianKernalRadius(Settings::BoundaryExtractor_CudaGaussianKernalRadius.value());
    m_boundaryExtractor->setCudaClassifyKernalRadius(Settings::BoundaryExtractor_CudaClassifyKernalRadius.value());
    m_boundaryExtractor->setCudaClassifyDistance(Settings::BoundaryExtractor_CudaClassifyDistance.value());
    m_boundaryExtractor->setCudaPeakClusterTolerance(Settings::BoundaryExtractor_CudaPeakClusterTolerance.intValue());
    m_boundaryExtractor->setCudaMinClusterPeaks(Settings::BoundaryExtractor_CudaMinClusterPeaks.intValue());
    m_boundaryExtractor->setCudaMaxClusterPeaks(Settings::BoundaryExtractor_CudaMaxClusterPeaks.intValue());
    m_boundaryExtractor->setCudaCornerHistSigma(Settings::BoundaryExtractor_CudaCornerHistSigma.value());

    m_cloudViewer->visualizer()->removeAllPointClouds();
    m_cloudViewer->visualizer()->removeAllShapes();
    //m_projectedCloudViewer->visualizer()->removeAllPointClouds();
    //m_projectedCloudViewer->visualizer()->removeAllShapes();
    //m_planeViewer->visualizer()->removeAllPointClouds();
    //m_planeViewer->visualizer()->removeAllShapes();
}

void ToolWindowBoundaryExtractor::onActionComputeGPU()
{
    init();

    int frameIndex = m_ui->comboBoxFrameIndex->currentIndex();
    Frame frame = m_device->getFrame(frameIndex);
    initDebugPixels(frame);
    
    if (!m_init)
    {
        m_frameGpu.parameters.colorWidth = frame.getColorWidth();
        m_frameGpu.parameters.colorHeight = frame.getColorHeight();
        m_frameGpu.parameters.depthWidth = frame.getDepthWidth();
        m_frameGpu.parameters.depthHeight = frame.getDepthHeight();
        m_frameGpu.allocate();
        m_init = true;
    }

    //cv::cuda::GpuMat colorMatGpu(frame.getColorHeight(), frame.getColorWidth(), CV_8UC3, m_frameGpu.colorImage);
    cv::cuda::GpuMat depthMatGpu(frame.getDepthHeight(), frame.getDepthWidth(), CV_16U, m_frameGpu.depthImage);
    //colorMatGpu.upload(frame.colorMat());
    depthMatGpu.upload(frame.depthMat());

    m_frameGpu.parameters.debugX = m_ui->comboBoxDebugX->currentIndex();
    m_frameGpu.parameters.debugY = m_ui->comboBoxDebugY->currentIndex();
    m_allBoundary = m_boundaryExtractor->computeCUDA(m_frameGpu);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = m_boundaryExtractor->cloud();
    pcl::PointCloud<pcl::Normal>::Ptr normals = m_boundaryExtractor->normals();
    m_boundaryPoints = m_boundaryExtractor->boundaryPoints();
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_veilPoints = m_boundaryExtractor->veilPoints();
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_borderPoints = m_boundaryExtractor->borderPoints();
    pcl::PointCloud<pcl::PointXYZI>::Ptr cornerPoints = m_boundaryExtractor->cornerPoints();

    qDebug() << "total points:" << cloud->size() << ", total boundary points:" << m_allBoundary->size() << ", border points:" << m_borderPoints->size() << ", veil points:" << m_veilPoints->size() << ", boundary points:" << m_boundaryPoints->size();

    //cv::Mat boundaryImage = (m_boundaryExtractor->boundaryMat() - 3) * 255;
    cv::Mat boundaryImage = m_boundaryExtractor->boundaryMat() * 60;
    cv::Mat rgbBoundaryImage;
    //cv::applyColorMap(boundaryImage, rgbBoundaryImage, cv::COLORMAP_JET);
    cv::cvtColor(boundaryImage, rgbBoundaryImage, cv::COLOR_GRAY2BGR);
    cv::Rect rect(m_frameGpu.parameters.debugX - 20, m_frameGpu.parameters.debugY - 20, 41, 41);
    cv::rectangle(rgbBoundaryImage, rect, cv::Scalar(0, 0, 255), 2);
    m_depthViewer2->setImage(cvMat2QImage(rgbBoundaryImage, true));

    if (m_boundaryPoints) {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> rColor(m_boundaryPoints, 255, 0, 0);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> gColor(m_veilPoints, 0, 255, 0);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> bColor(m_borderPoints, 0, 0, 255);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> yColor(cornerPoints, 255, 255, 0);

        m_cloudViewer->visualizer()->addPointCloud(m_boundaryPoints, rColor, "boundary points");
        m_cloudViewer->visualizer()->addPointCloud(m_veilPoints, gColor, "veil points");
        m_cloudViewer->visualizer()->addPointCloud(m_borderPoints, bColor, "border points");
        m_cloudViewer->visualizer()->addPointCloud(cornerPoints, yColor, "corner points");
        m_cloudViewer->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "boundary points");
        m_cloudViewer->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "veil points");
        m_cloudViewer->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "border points");
        m_cloudViewer->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "corner points");
    }

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorCloud;
    pcl::IndicesPtr indices(new std::vector<int>);
    colorCloud = frame.getCloud(*indices);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgbHandler(colorCloud);
    m_cloudViewer->visualizer()->addPointCloud(colorCloud, rgbHandler, "scene cloud");

    //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> hColor(cloud, 64, 64, 64);
    //m_cloudViewer->visualizer()->addPointCloud(cloud, hColor, "scene cloud");
    m_cloudViewer->visualizer()->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(colorCloud, normals, 100, 0.1f);

    StopWatch::instance().debugPrint();
}

void ToolWindowBoundaryExtractor::onActionComputeVBRG()
{
    init();

    int frameIndex = m_ui->comboBoxFrameIndex->currentIndex();
    Frame frame = m_device->getFrame(frameIndex);
    pcl::IndicesPtr indices(new std::vector<int>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorCloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    colorCloud = frame.getCloud(*indices);
    pcl::copyPointCloud<pcl::PointXYZRGB, pcl::PointXYZ>(*colorCloud, *cloud);
    m_boundaryExtractor->setInputCloud(cloud);
    TICK("VBRG");
    pcl::PointCloud<pcl::PointXYZI>::Ptr classifiedCloud = m_boundaryExtractor->computeVBRG();
    TOCK("VBRG");
    qDebug() << "classified cloud size:" << classifiedCloud->size();

    pcl::PointCloud<pcl::PointXYZ>::Ptr voxelCloud = m_boundaryExtractor->voxelPoints();
    pcl::PointCloud<pcl::Normal>::Ptr normals = m_boundaryExtractor->normals();
    QMap<qulonglong, VoxelInfo> voxelInfos = m_boundaryExtractor->voxelInfos();

    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> iColor(classifiedCloud, "intensity");
    m_cloudViewer->visualizer()->addPointCloud(classifiedCloud, iColor, "classified cloud");
    //pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgbColor(colorCloud);
    //m_cloudViewer->visualizer()->addPointCloud(colorCloud, rgbColor, "scene cloud");
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> yColor(voxelCloud, 255, 0, 0);
    m_cloudViewer->visualizer()->addPointCloud(voxelCloud, yColor, "voxel cloud");
    m_cloudViewer->visualizer()->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(voxelCloud, normals, 1, 0.1f);
    m_cloudViewer->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "voxel cloud");

    for (QMap<qulonglong, VoxelInfo>::iterator i = voxelInfos.begin(); i != voxelInfos.end(); i++)
    {
        VoxelInfo& info = i.value();
        Eigen::Vector3f min = info.min;
        Eigen::Vector3f max = info.max;
        //qDebug() << info.nodeId << min.x() << min.y() << min.z() << max.x() << max.y() << max.z();
        QString cubeName = QString("cube_%1").arg(info.nodeId);
        m_cloudViewer->visualizer()->addCube(min.x(), max.x(), min.y(), max.y(), min.z(), max.z(), 0.5, 0.5, 0.5, cubeName.toStdString());
    }
    m_cloudViewer->visualizer()->setRepresentationToWireframeForAllActors();
    StopWatch::instance().debugPrint();
}

void ToolWindowBoundaryExtractor::onActionSave()
{
    QString fileName = QString("%1/%2_%3_boundary.pcd").arg(QDir::currentPath()).arg(Settings::SensorReader_SamplePath.value()).arg(m_ui->comboBoxFrameIndex->currentIndex());
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
    //PARAMETERS.setValue("downsampling_method", m_ui->comboBoxDownsamplingMethod->currentIndex(), "BoundaryExtractor");
    //PARAMETERS.setValue("enable_removal_filter", m_ui->checkBoxEnableRemovalFilter->isChecked(), "BoundaryExtractor");
    //PARAMETERS.setValue("downsample_leaf_size", m_ui->doubleSpinBoxDownsampleLeafSize->value(), "BoundaryExtractor");
    //PARAMETERS.setValue("outlier_removal_mean_k", m_ui->doubleSpinBoxOutlierRemovalMeanK->value(), "BoundaryExtractor");
    //PARAMETERS.setValue("std_dev_mul_thresh", m_ui->doubleSpinBoxStddevMulThresh->value(), "BoundaryExtractor");
    /*PARAMETERS.setValue("gaussian_sigma", m_ui->doubleSpinBoxGaussianSigma->value(), "BoundaryExtractor");
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
    PARAMETERS.setValue("plane_distance_threshold", m_ui->doubleSpinBoxPlaneDistanceThreshold->value(), "BoundaryExtractor");*/

    //PARAMETERS.save();
}

void ToolWindowBoundaryExtractor::initDebugPixels(Frame& frame)
{
    int xIndex = m_ui->comboBoxDebugX->currentIndex();
    int yIndex = m_ui->comboBoxDebugY->currentIndex();
    if (xIndex < 0) xIndex = 300;
    if (yIndex < 0) yIndex = 262;
    m_ui->comboBoxDebugX->clear();
    m_ui->comboBoxDebugY->clear();
    for (int i = 0; i < frame.getDepthWidth(); i++)
    {
        m_ui->comboBoxDebugX->addItem(QString::number(i));
    }
    for (int i = 0; i < frame.getDepthHeight(); i++)
    {
        m_ui->comboBoxDebugY->addItem(QString::number(i));
    }
    m_ui->comboBoxDebugX->setCurrentIndex(xIndex);
    m_ui->comboBoxDebugY->setCurrentIndex(yIndex);
}

void ToolWindowBoundaryExtractor::onActionLoadDataSet()
{
    m_device.reset(Device::createDevice());
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
