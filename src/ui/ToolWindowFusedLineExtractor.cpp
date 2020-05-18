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
    m_ui->widgetImageAnchor->setImage(cvMat2QImage(m_extractor->colorLinesMat()));
    m_ui->widgetImageAngle->setImage(cvMat2QImage(m_extractor->linesMat()));
    
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


    if (m_ui->checkBoxShowPoints->isChecked())
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr beCloud = m_extractor->allBoundary();
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> h1(beCloud, "intensity");
        m_cloudViewer->visualizer()->addPointCloud(beCloud, h1, "cloud_src");

        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> h2(cloud, 255, 0, 0);
        m_cloudViewer->visualizer()->addPointCloud(cloud, h2, cloudName.toStdString());
        m_cloudViewer->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, cloudName.toStdString());
    }

    m_cloudViewer->visualizer()->removeAllShapes();
    //QMap<int, LineSegment> lines = m_extractor->lines();
    pcl::PointCloud<LineSegment>::Ptr linesCloud = m_extractor->linesCloud();
    //for (QMap<int, LineSegment>::iterator i = lines.begin(); i != lines.end(); i++)
    for (int i = 0; i < linesCloud->points.size(); i++)
    {
        QString lineNo = QString("line_%1").arg(i);
        QString textNo = QString("%1").arg(i);
        LineSegment line = linesCloud->points[i];
        pcl::PointXYZ start, end, middle;
        start.getVector3fMap() = line.start();
        end.getVector3fMap() = line.end();
        middle.getVector3fMap() = line.middle();
        m_cloudViewer->visualizer()->addLine(start, end, i, 255, 255, lineNo.toStdString());
        m_cloudViewer->visualizer()->addText3D(textNo.toStdString(), middle, 0.01, 1.0, 0.0, 0.0, textNo.toStdString());
    }
}

void ToolWindowFusedLineExtractor::updateWidgets()
{
    m_ui->actionLoad_Data_Set->setEnabled(!m_isLoaded);
    m_ui->actionCompute->setEnabled(m_isLoaded);
}