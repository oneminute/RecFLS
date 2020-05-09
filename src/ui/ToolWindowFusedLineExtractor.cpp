#include "ToolWindowFusedLineExtractor.h"
#include "ui_ToolWindowFusedLineExtractor.h"

#include <QDebug>
#include <QtMath>
#include <QPushButton>
#include <QFile>
#include <QTextStream>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/pca.h>

#include "device/SensorReaderDevice.h"
#include "extractor/FusedLineExtractor.h"
#include "common/Parameters.h"
#include "util/Utils.h"
#include "util/StopWatch.h"
#include "cuda/cuda.hpp"


ToolWindowFusedLineExtractor::ToolWindowFusedLineExtractor(QWidget *parent) 
    : QMainWindow(parent)
    , m_ui(new Ui::ToolWindowFusedLineExtractor)
    , m_isInit(false)
    , m_isLoaded(false)
{
    m_ui->setupUi(this);

    m_cloudViewer = new CloudViewer;

    m_cloudViewer->setCameraPosition(0, 0, -1.5f, 0, 0, 0, 0, -1, 0);

    m_ui->horizontalLayoutCenter->addWidget(m_cloudViewer);

    connect(m_ui->actionLoad_Data_Set, &QAction::triggered, this, &ToolWindowFusedLineExtractor::onActionLoadDataSet);
    connect(m_ui->actionCompute, &QAction::triggered, this, &ToolWindowFusedLineExtractor::onActionCompute);
    connect(m_ui->pushButtonShowPoints, &QPushButton::clicked, this, &ToolWindowFusedLineExtractor::onActionShowPoints);

    updateWidgets();
}

ToolWindowFusedLineExtractor::~ToolWindowFusedLineExtractor()
{
    if (m_isInit)
    {
        m_frameGpu.free();
        m_frameBEGpu.free();
    }
}

void ToolWindowFusedLineExtractor::initCompute()
{
    m_extractor.reset(new FusedLineExtractor);

    int frameIndex = m_ui->comboBoxFrame->currentIndex();
    m_frame = m_device->getFrame(frameIndex);

    m_ui->widgetImage->setImage(cvMat2QImage(m_frame.colorMat()));

    if (!m_isInit)
    {
        cuda::FusedLineParameters parameters;
        parameters.cx = m_device->cx();
        parameters.cy = m_device->cy();
        parameters.fx = m_device->fx();
        parameters.fy = m_device->fy();
        parameters.minDepth = Settings::BoundaryExtractor_MinDepth.value();
        parameters.maxDepth = Settings::BoundaryExtractor_MaxDepth.value();
        parameters.depthShift = m_device->depthShift();
        parameters.rgbWidth = m_frame.getColorWidth();
        parameters.rgbHeight = m_frame.getColorHeight();
        parameters.normalKernalRadius = Settings::ICPMatcher_CudaNormalKernalRadius.intValue();
        parameters.normalKnnRadius = Settings::ICPMatcher_CudaNormalKnnRadius.value();
        parameters.depthWidth = m_frame.getDepthWidth();
        parameters.depthHeight = m_frame.getDepthHeight();
        parameters.blockSize = Settings::ICPMatcher_CudaBlockSize.intValue();
        parameters.gradientThreshold = 20;
        parameters.searchRadius = 4;

        m_frameGpu.parameters = parameters;
        m_frameGpu.allocate();

        m_frameBEGpu.parameters.colorWidth = m_frame.getColorWidth();
        m_frameBEGpu.parameters.colorHeight = m_frame.getColorHeight();
        m_frameBEGpu.parameters.depthWidth = m_frame.getDepthWidth();
        m_frameBEGpu.parameters.depthHeight = m_frame.getDepthHeight();
        m_frameBEGpu.allocate();

        m_isInit = true;
    }

    m_frameGpu.rgbMatGpu.upload(m_frame.colorMat());
    m_frameGpu.depthMatGpu.upload(m_frame.depthMat());

    cv::cuda::GpuMat colorMatGpu(m_frame.getColorHeight(), m_frame.getColorWidth(), CV_8UC3, m_frameBEGpu.colorImage);
    cv::cuda::GpuMat depthMatGpu(m_frame.getDepthHeight(), m_frame.getDepthWidth(), CV_16U, m_frameBEGpu.depthImage);
    colorMatGpu.upload(m_frame.colorMat());
    depthMatGpu.upload(m_frame.depthMat());

    //TICK("generate_cloud");
    //cuda::icpGenerateCloud(m_frameSrc);
    //cuda::icpGenerateCloud(m_frameDst);
    //m_cache.srcCloud = m_frameSrc.pointCloud;
    //m_cache.dstCloud = m_frameDst.pointCloud;
    //m_cache.srcNormals = m_frameSrc.pointCloudNormals;
    //m_cache.dstNormals = m_frameDst.pointCloudNormals;
    //TOCK("generate_cloud");

    //TICK("icp_cloud_downloading");
    //std::vector<float3> pointsSrc;
    //m_frameSrc.pointCloud.download(pointsSrc);
    //std::vector<float3> normalsSrc;
    //m_frameSrc.pointCloudNormals.download(normalsSrc);
    //std::vector<float3> pointsDst;
    //m_frameDst.pointCloud.download(pointsDst);
    //std::vector<float3> normalsDst;
    //m_frameDst.pointCloudNormals.download(normalsDst);

    //m_cloudSrc.reset(new pcl::PointCloud<pcl::PointXYZI>);
    //m_cloudDst.reset(new pcl::PointCloud<pcl::PointXYZI>);
    //m_normalsSrc.reset(new pcl::PointCloud<pcl::Normal>);
    //m_normalsDst.reset(new pcl::PointCloud<pcl::Normal>);
    //for(int i = 0; i < frameSrc.getDepthHeight(); i++) 
    //{
    //    for(int j = 0; j < frameSrc.getDepthWidth(); j++) 
    //    {
    //        int index = i * frameSrc.getDepthWidth() + j;

    //        Eigen::Vector3f valueSrc = toVector3f(pointsSrc[index]);
    //        Eigen::Vector3f valueDst = toVector3f(pointsDst[index]);
    //        pcl::PointXYZI ptSrc;
    //        pcl::Normal normalSrc;
    //        pcl::PointXYZI ptDst;
    //        pcl::Normal normalDst;

    //        if (!(isnan(valueSrc.x()) || isnan(valueSrc.y()) || isnan(valueSrc.z())))
    //        {
    //            ptSrc.getVector3fMap() = valueSrc;
    //            ptSrc.intensity = index;
    //            normalSrc.getNormalVector3fMap() = toVector3f(normalsSrc[index]);

    //            m_cloudSrc->points.push_back(ptSrc);
    //            m_normalsSrc->points.push_back(normalSrc);
    //        }

    //        if (!(isnan(valueDst.x()) || isnan(valueDst.y()) || isnan(valueDst.z())))
    //        {
    //            ptDst.getVector3fMap() = valueDst;
    //            ptDst.intensity = index;
    //            normalDst.getNormalVector3fMap() = toVector3f(normalsDst[index]);

    //            m_cloudDst->points.push_back(ptDst);
    //            m_normalsDst->points.push_back(normalDst);
    //        }
    //    }
    //}

    //m_cloudSrc->width = m_cloudSrc->points.size();
    //m_cloudSrc->height = 1;
    //m_cloudSrc->is_dense = true;

    //m_cloudDst->width = m_cloudDst->points.size();
    //m_cloudDst->height = 1;
    //m_cloudDst->is_dense = true;

    //m_normalsSrc->width = m_normalsSrc->points.size();
    //m_normalsSrc->height = 1;
    //m_normalsSrc->is_dense = true;

    //m_normalsDst->width = m_normalsDst->points.size();
    //m_normalsDst->height = 1;
    //m_normalsDst->is_dense = true;
    //TOCK("icp_cloud_downloading");

    //m_tree.reset(new pcl::search::KdTree<pcl::PointXYZI>);
    //m_tree->setInputCloud(m_cloudDst);

    //m_rotationDelta = Eigen::Matrix3f::Identity();
    //m_translationDelta = Eigen::Vector3f::Zero();
    //m_rotation = Eigen::Matrix3f::Identity();
    //m_translation = Eigen::Vector3f::Zero();
    //m_pose = Eigen::Matrix4f::Identity();
    //m_iteration = 0;

    //// ÏÔÊ¾µãÔÆ
    //{
    //    m_cloudViewer->removeAllClouds();
    //    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> h1(m_cloudSrc, 255, 0, 0);
    //    m_cloudViewer->visualizer()->addPointCloud(m_cloudSrc, h1, "cloud_src");
    //    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> h2(m_cloudDst, 0, 0, 255);
    //    m_cloudViewer->visualizer()->addPointCloud(m_cloudDst, h2, "cloud_dst");
    //    //m_cloudViewer->visualizer()->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(m_cloudSrc, m_normalsSrc, 100, 0.1f);
    //    //m_cloudViewer->visualizer()->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(m_cloudDst, m_normalsDst, 100, 0.1f);
    //}

    m_isInit = true;
}

void ToolWindowFusedLineExtractor::compute()
{
    initCompute();

    m_cloudViewer->removeAllClouds();
    m_cloudViewer->visualizer()->removeAllShapes();

    //m_extractor->computeGPU(m_frameGpu);
    m_extractor->compute(m_frame, m_frameBEGpu);
    pcl::PointCloud<pcl::PointXYZI>::Ptr beCloud = m_extractor->allBoundary();
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> beh(beCloud, "intensity");
    m_cloudViewer->visualizer()->addPointCloud(beCloud, beh, "cloud_src");

    QMap<int, pcl::PointCloud<pcl::PointXYZI>::Ptr>& groupPoints = m_extractor->groupPoints();
    m_ui->comboBoxGroupPoints->clear();
    for (QMap<int, pcl::PointCloud<pcl::PointXYZI>::Ptr>::iterator i = groupPoints.begin(); i != groupPoints.end(); i++)
    {
        int index = i.key();
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud = i.value();

        if (cloud->size() < 10)
            continue;

        m_ui->comboBoxGroupPoints->addItem(QString("%1 %2").arg(index).arg(cloud->size()), index);
    }

    

    /*cv::Mat grayImage;
    m_frameGpu.grayMatGpu.download(grayImage);
    m_ui->widgetImageGaussian->setImage(cvMat2QImage(grayImage, false));

    cv::Mat gradientImage;
    m_frameGpu.gradientMatGpu.download(gradientImage);
    cv::Mat result8UC1;
    cv::convertScaleAbs(gradientImage, result8UC1);
    m_ui->widgetImageGradient->setImage(cvMat2QImage(result8UC1, false));

    cv::Mat anchorImage;
    m_frameGpu.anchorMatGpu.download(anchorImage);
    m_ui->widgetImageAnchor->setImage(cvMat2QImage(anchorImage, false));

    cv::Mat angleImage;
    m_frameGpu.angleMatGpu.download(angleImage);
    cv::convertScaleAbs(angleImage, result8UC1);
    m_ui->widgetImageAngle->setImage(cvMat2QImage(angleImage, false, uCvQtDepthBlueToRed));

    cv::Mat radiusImage;
    m_frameGpu.radiusMatGpu.download(radiusImage);
    m_ui->widgetImageRadius->setImage(cvMat2QImage(radiusImage, false));*/

    //showMatchedClouds();
    //updateWidgets();

    StopWatch::instance().debugPrint();
}

void ToolWindowFusedLineExtractor::onActionLoadDataSet()
{
    m_device.reset(new SensorReaderDevice);
    if (!m_device->open())
    {
        qDebug() << "Open device failed.";
        return;
    }

    m_ui->comboBoxFrame->clear();
    for (int i = 0; i < m_device->totalFrames(); i++)
    {
        m_ui->comboBoxFrame->addItem(QString::number(i));
    }
    m_ui->comboBoxFrame->setCurrentIndex(0);


    m_isLoaded = true;
    updateWidgets();
}

void ToolWindowFusedLineExtractor::onActionCompute()
{
    compute();
}

void ToolWindowFusedLineExtractor::onActionShowPoints()
{
    int cloudIndex = m_ui->comboBoxGroupPoints->currentData().toInt();
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud = m_extractor->groupPoints()[cloudIndex];

    QString cloudName = QString("cloud_%1").arg(cloudIndex);
    m_cloudViewer->visualizer()->removeAllPointClouds();

    pcl::PointCloud<pcl::PointXYZI>::Ptr beCloud = m_extractor->allBoundary();
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> h1(beCloud, "intensity");
    m_cloudViewer->visualizer()->addPointCloud(beCloud, h1, "cloud_src");

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> h2(cloud, 255, 0, 0);
    m_cloudViewer->visualizer()->addPointCloud(cloud, h2, cloudName.toStdString());
    m_cloudViewer->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, cloudName.toStdString());

    m_cloudViewer->visualizer()->removeAllShapes();
    QMap<int, LS3D> lines = m_extractor->lines();
    for (QMap<int, LS3D>::iterator i = lines.begin(); i != lines.end(); i++)
    {
        QString lineNo = QString("line_%1").arg(i.key());
        QString textNo = QString("%1").arg(i.key());
        LS3D line = i.value();
        m_cloudViewer->visualizer()->addLine(line.start, line.end, i.key() * 10, 255, 255, lineNo.toStdString());
        m_cloudViewer->visualizer()->addText3D(textNo.toStdString(), line.center, 0.01, 1.0, 0.0, 0.0, textNo.toStdString());
    }
}

void ToolWindowFusedLineExtractor::updateWidgets()
{
    m_ui->actionLoad_Data_Set->setEnabled(!m_isLoaded);
    m_ui->actionCompute->setEnabled(m_isLoaded);
}