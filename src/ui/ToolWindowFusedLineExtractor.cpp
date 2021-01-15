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
#include "device/IclNuimDevice.h"
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

	//m_cloudViewer->visualizer()->setBackgroundColor(255, 255, 255);
    m_cloudViewer->setCameraPosition(0, 0, -1.5f, 0, 0, 0, 0, -1, 0);

    m_ui->horizontalLayoutCenter->addWidget(m_cloudViewer);

	m_ui->widgetImageAnchor->setBackgroundColor(QColor(255, 255, 255));
	

    connect(m_ui->actionLoad_Data_Set, &QAction::triggered, this, &ToolWindowFusedLineExtractor::onActionLoadDataSet);
    connect(m_ui->actionCompute, &QAction::triggered, this, &ToolWindowFusedLineExtractor::onActionCompute);
    connect(m_ui->pushButtonShowPoints, &QPushButton::clicked, this, &ToolWindowFusedLineExtractor::onActionShowPoints);

    updateWidgets();
}

ToolWindowFusedLineExtractor::~ToolWindowFusedLineExtractor()
{
    if (m_isInit)
    {
    }
}

void ToolWindowFusedLineExtractor::initCompute()
{
    m_extractor.reset(new FusedLineExtractor);

    int frameIndex = m_ui->comboBoxFrame->currentIndex();
    m_frame = m_device->getFrame(frameIndex);

    m_ui->widgetImage->setImage(cvMat2QImage(m_frame.colorMat()));

    m_isInit = true;
}

void ToolWindowFusedLineExtractor::compute()
{
    initCompute();

    m_cloudViewer->removeAllClouds();
    m_cloudViewer->visualizer()->removeAllShapes();

    //m_extractor->computeGPU(m_frameGpu);
    m_flFrame = m_extractor->compute(m_frame);
	//m_extractor->generateVoxelsDescriptors(m_frame, m_flFrame.lines(), 0.25f, 5, 4, 8, m_frame.getColorWidth(), m_frame.getColorHeight(), 
        //m_frame.getDevice()->cx(), m_frame.getDevice()->cy(), m_frame.getDevice()->fx(), m_frame.getDevice()->fy());
    //m_extractor->generateCylinderDescriptors(m_frame, m_flFrame.lines(), 0.5f, 5, m_frame.getColorWidth(), m_frame.getColorHeight(),
        //m_frame.getDevice()->cx(), m_frame.getDevice()->cy(), m_frame.getDevice()->fx(), m_frame.getDevice()->fy());
    //pcl::PointCloud<pcl::PointXYZINormal>::Ptr beCloud = m_extractor->allBoundary();
    //pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZINormal> beh(beCloud, "intensity");
    //m_cloudViewer->visualizer()->addPointCloud(beCloud, beh, "cloud_src");

    QMap<int, pcl::PointCloud<pcl::PointXYZINormal>::Ptr>& groupPoints = m_extractor->groupPoints();
    m_ui->comboBoxGroupPoints->clear();
    for (QMap<int, pcl::PointCloud<pcl::PointXYZINormal>::Ptr>::iterator i = groupPoints.begin(); i != groupPoints.end(); i++)
    {
        int index = i.key();
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud = i.value();

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
    //m_device.reset(new SensorReaderDevice);
    m_device.reset(Device::createDevice());
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
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud = m_extractor->groupPoints()[cloudIndex];

    QString cloudName = QString("cloud_%1").arg(cloudIndex);
    m_cloudViewer->visualizer()->removeAllPointClouds();

    /*if (m_ui->checkBoxShowPoints->isChecked())
    {
        pcl::PointCloud<pcl::PointXYZINormal>::Ptr beCloud = m_extractor->allBoundary();
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZINormal> h1(beCloud, "intensity");
        m_cloudViewer->visualizer()->addPointCloud(beCloud, h1, "cloud_src");

        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZINormal> h2(cloud, 255, 0, 0);
        m_cloudViewer->visualizer()->addPointCloud(cloud, h2, cloudName.toStdString());
        m_cloudViewer->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, cloudName.toStdString());
    }*/

    m_cloudViewer->visualizer()->removeAllShapes();
    //QMap<int, LineSegment> lines = m_extractor->lines();
    pcl::PointCloud<LineSegment>::Ptr linesCloud = m_flFrame.lines();
    //for (QMap<int, LineSegment>::iterator i = lines.begin(); i != lines.end(); i++)
    Eigen::Vector3f minPoint = m_extractor->minPoint();
    for (int i = 0; i < linesCloud->points.size(); i++)
    {
        QColor color(QColor::Hsv);
        color.setHsv(i * 359 / linesCloud->points.size(), 255, 255);
        QColor rgb = color.convertTo(QColor::Rgb);
        QString lineNo = QString("line_%1").arg(i);
        QString textNo = QString("%1").arg(i);
        LineSegment line = linesCloud->points[i];
        pcl::PointXYZ start, end, middle;
        start.getVector3fMap() = line.start();
        end.getVector3fMap() = line.end();
        middle.getVector3fMap() = line.middle();
        m_cloudViewer->visualizer()->addLine(start, end, 1.0, 1.0, 1.0, lineNo.toStdString());
        if (i == cloudIndex)
            m_cloudViewer->visualizer()->addText3D(textNo.toStdString(), middle, 0.1, 1.0, 1.0, 1.0, textNo.toStdString());
        else
            m_cloudViewer->visualizer()->addText3D(textNo.toStdString(), middle, 0.025, 0.0, 1.0, 0.0, textNo.toStdString());
		m_cloudViewer->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, lineNo.toStdString());

        QString cylinderCloudName = QString("cylinder_cloud_%1").arg(i);
        //pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZINormal> handle(line.cylinderCloud(), "intensity");
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZINormal> handle(line.cylinderCloud(), rgb.red(), rgb.green(), rgb.blue());
        //m_cloudViewer->visualizer()->addPointCloud(line.cylinderCloud(), handle, cylinderCloudName.toStdString());
        //m_cloudViewer->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, cylinderCloudName.toStdString());

        if (i != cloudIndex)
            continue;

        /*std::vector<std::vector<Eigen::Vector3f>> voxels = line.lineCylinders();
        for (int li = 0; li < voxels.size(); li++)
        {
            QColor color = QColor::fromHsl(li * 360 / voxels.size(), 255, 255);
            qDebug() << lineNo << li << voxels[li].size();
            for (int vi = 0; vi < voxels[li].size(); vi++)
            {

                Eigen::Vector3f voxelKey = voxels[li][vi];
                Eigen::Vector3f rectMin = voxelKey * m_extractor->resolution() + minPoint;
                Eigen::Vector3f rectMax = rectMin + Eigen::Vector3f(m_extractor->resolution(), m_extractor->resolution(), m_extractor->resolution());
                QString cubeName = QString("cube_%1_%2_%3").arg(i).arg(li).arg(vi);
                m_cloudViewer->visualizer()->addCube(rectMin.x(), rectMax.x(), rectMin.y(), rectMax.y(), rectMin.z(), rectMax.z(), 0, 0, 1, cubeName.toStdString());
                m_cloudViewer->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.1, cubeName.toStdString());
            }
        }*/
    }
}

void ToolWindowFusedLineExtractor::updateWidgets()
{
    m_ui->actionLoad_Data_Set->setEnabled(!m_isLoaded);
    m_ui->actionCompute->setEnabled(m_isLoaded);
}