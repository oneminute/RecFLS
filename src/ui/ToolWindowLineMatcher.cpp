#include "ToolWindowLineMatcher.h"
#include "ui_ToolWindowLineMatcher.h"

#include <QDebug>
#include <QtMath>
#include <QPushButton>
#include <QElapsedTimer>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization//impl/point_cloud_geometry_handlers.hpp>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/common/pca.h>
#include <pcl/visualization/impl/point_cloud_geometry_handlers.hpp>

#include "common/Parameters.h"
#include "util/Utils.h"


ToolWindowLineMatcher::ToolWindowLineMatcher(QWidget *parent) 
    : QMainWindow(parent)
    , m_ui(new Ui::ToolWindowLineMatcher)
    , m_isStepMode(false)
    , m_isInit(false)
    , m_isLoaded(false)
    , m_iteration(0)
    //, m_diameter1(0)
    //, m_diameter2(0)
    , m_error(0)
{
    m_ui->setupUi(this);

    m_cloudViewer1 = new CloudViewer;
    m_cloudViewer2 = new CloudViewer;
    m_cloudViewer3 = new CloudViewer;
	//m_cloudViewer1->visualizer()->setBackgroundColor(255, 255, 255);
	//m_cloudViewer2->visualizer()->setBackgroundColor(255, 255, 255);
	//m_cloudViewer3->visualizer()->setBackgroundColor(255, 255, 255);

    m_cloudViewer1->setCameraPosition(0, 0, -1.5f, 0, 0, 0, 0, -1, 0);
    m_cloudViewer2->setCameraPosition(0, 0, -1.5f, 0, 0, 0, 0, -1, 0);
    m_cloudViewer3->setCameraPosition(0, 0, -1.5f, 0, 0, 0, 0, -1, 0);

    m_ui->verticalLayoutCloud1->addWidget(m_cloudViewer1);
    m_ui->verticalLayoutCloud2->addWidget(m_cloudViewer2);
    m_ui->verticalLayoutCloud3->addWidget(m_cloudViewer3);
	

    connect(m_ui->actionLoad_Data_Set, &QAction::triggered, this, &ToolWindowLineMatcher::onActionLoadDataSet);
    connect(m_ui->actionMatch, &QAction::triggered, this, &ToolWindowLineMatcher::onActionMatch);
    connect(m_ui->actionMatch_Gpu, &QAction::triggered, this, &ToolWindowLineMatcher::onActionMatchGpu);
    connect(m_ui->actionBegin_Step, &QAction::triggered, this, &ToolWindowLineMatcher::onActionBeginStep);
    connect(m_ui->actionStep, &QAction::triggered, this, &ToolWindowLineMatcher::onActionStep);
    connect(m_ui->actionReset, &QAction::triggered, this, &ToolWindowLineMatcher::onActionReset);

    connect(m_ui->comboBoxDstFrame, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &ToolWindowLineMatcher::onComboBox1CurrentIndexChanged);
    connect(m_ui->pushButtonShowLineChainPair, &QPushButton::clicked, this, &ToolWindowLineMatcher::onActionShowPair);

    updateWidgets();
}

ToolWindowLineMatcher::~ToolWindowLineMatcher()
{
    if (m_isInit)
    {
        //m_frameGpuSrc.free();
        //m_frameGpuDst.free();
        m_frameGpuBESrc.free();
        m_frameGpuBEDst.free();
    }
}

void ToolWindowLineMatcher::initCompute()
{
    if (!m_lineExtractor)
    {
        m_lineExtractor.reset(new FusedLineExtractor);
    }

    if (!m_lineMatcher)
    {
        m_lineMatcher.reset(new LineMatcher);
    }

    int frameIndexDst = m_ui->comboBoxDstFrame->currentIndex();
    Frame frameDst = m_device->getFrame(frameIndexDst);
    int frameIndexSrc = m_ui->comboBoxSrcFrame->currentIndex();
    Frame frameSrc = m_device->getFrame(frameIndexSrc);

    m_ui->widgetFrame1->setImage(cvMat2QImage(frameSrc.colorMat()));
    m_ui->widgetFrame2->setImage(cvMat2QImage(frameDst.colorMat()));

    pcl::IndicesPtr indicesSrc(new std::vector<int>);
    pcl::IndicesPtr indicesDst(new std::vector<int>);
    m_cloudSrc.reset(new pcl::PointCloud<pcl::PointXYZ>);
    m_cloudDst.reset(new pcl::PointCloud<pcl::PointXYZ>);

    m_colorCloudSrc = frameSrc.getCloud(*indicesSrc);
    m_colorCloudDst = frameDst.getCloud(*indicesDst);
    pcl::copyPointCloud<pcl::PointXYZRGB, pcl::PointXYZ>(*m_colorCloudSrc, *m_cloudSrc);
    pcl::copyPointCloud<pcl::PointXYZRGB, pcl::PointXYZ>(*m_colorCloudDst, *m_cloudDst);

    if (!m_isInit)
    {
        cuda::Parameters beParameters;
        beParameters.colorWidth = frameSrc.getColorWidth();
        beParameters.colorHeight = frameSrc.getColorHeight();
        beParameters.depthWidth = frameSrc.getDepthWidth();
        beParameters.depthHeight = frameSrc.getDepthHeight();

        m_frameGpuBESrc.parameters = beParameters;
        m_frameGpuBESrc.allocate();
        m_frameGpuBEDst.parameters = beParameters;
        m_frameGpuBEDst.allocate();

        m_isInit = true;
    }

    cv::cuda::GpuMat depthMatGpuSrc(frameSrc.getDepthHeight(), frameSrc.getDepthWidth(), CV_16U, m_frameGpuBESrc.depthImage);
    depthMatGpuSrc.upload(frameSrc.depthMat());

    cv::cuda::GpuMat depthMatGpuDst(frameDst.getDepthHeight(), frameDst.getDepthWidth(), CV_16U, m_frameGpuBEDst.depthImage);
    depthMatGpuDst.upload(frameDst.depthMat());
    
    m_flFrameSrc = m_lineExtractor->compute(frameSrc);
    m_beCloudSrc = m_lineExtractor->allBoundary();

    m_flFrameDst = m_lineExtractor->compute(frameDst);
    m_flFrameDst.setPose(Eigen::Matrix4f::Identity());
    m_beCloudDst = m_lineExtractor->allBoundary();

    m_cloudViewer1->visualizer()->removeAllPointClouds();
    m_cloudViewer1->visualizer()->removeAllShapes();
    m_cloudViewer2->visualizer()->removeAllPointClouds();
    m_cloudViewer2->visualizer()->removeAllShapes();
    m_cloudViewer3->visualizer()->removeAllPointClouds();
    m_cloudViewer3->visualizer()->removeAllShapes();

    m_tree.reset(new pcl::KdTreeFLANN<LineSegment>());
    m_tree->setInputCloud(m_flFrameDst.lines());

    Eigen::AngleAxisf rollAngle(M_PI / 72, Eigen::Vector3f::UnitZ());
    Eigen::AngleAxisf yawAngle(0, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf pitchAngle(0, Eigen::Vector3f::UnitX());
    Eigen::Quaternion<float> q = rollAngle * yawAngle * pitchAngle;
    Eigen::Matrix3f rotationMatrix = q.matrix();

    Eigen::Matrix3f rotation = Eigen::Matrix3f::Identity();
    Eigen::Vector3f translation = Eigen::Vector3f::Zero();
    m_pose = Eigen::Matrix4f::Identity();
    m_pose.topLeftCorner(3, 3) = rotation;
    m_pose.topRightCorner(3, 1) = translation;
    m_flFrameSrc.setPose(m_pose);

    m_iteration = 0;
    m_iterationDuration = 0;
    m_totalDuration = 0;

    qDebug() << "src lines size:" << m_flFrameSrc.lines()->size() << ", dst lines size:" << m_flFrameDst.lines()->size();
    m_lineMatcher->match(m_flFrameSrc.lines(), m_flFrameDst.lines(), m_tree, m_pairs, m_weights, m_pose);

   {
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZINormal> behSrc(m_beCloudSrc, "intensity");
        m_cloudViewer2->visualizer()->addPointCloud(m_beCloudSrc, behSrc, "cloud_src");
	
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZINormal> behDst(m_beCloudDst, "intensity");
        m_cloudViewer3->visualizer()->addPointCloud(m_beCloudDst, behDst, "cloud_dst");
		
    }

    showCloudAndLines(m_cloudViewer2, m_flFrameSrc.lines());
    showCloudAndLines(m_cloudViewer3, m_flFrameDst.lines());
    showMatchedClouds();

    m_isInit = true;
}

void ToolWindowLineMatcher::compute()
{
    initCompute();
    QElapsedTimer timer;
    timer.start();
    m_pose = m_lineMatcher->compute(m_flFrameSrc, m_flFrameDst, m_error, m_iteration);
    m_iterationDuration = timer.nsecsElapsed() / 1000000.f;
    m_totalDuration += m_iterationDuration;
    m_ui->comboBoxLineChainPairs->clear();
    
    showMatchedClouds();
    updateWidgets();
}

void ToolWindowLineMatcher::showCloudAndLines(CloudViewer* viewer, pcl::PointCloud<LineSegment>::Ptr& lines)
{
    for (int i = 0; i < lines->points.size()/* && errors[i]*/; i++)
    {
        LineSegment& line = lines->points[i];
        if (line.length() < 0.1f)
            continue;

        std::string lineNo = "line_" + std::to_string(line.index());
        std::string textNo = "text_" + std::to_string(line.index());
        pcl::PointXYZI start, end, middle;
        start.getVector3fMap() = line.start();
        end.getVector3fMap() = line.end();
        middle.getVector3fMap() = line.middle();
        Eigen::Vector3f dir = line.direction();
        //m_cloudViewer->visualizer()->addArrow(end, start, r, g, b, 0, lineNo);
        //std::cout << line.red() << ", " << line.green() << ", " << line.blue() << std::endl;
        //viewer->visualizer()->addLine(start, end, line.red() / 255, line.green() / 255, line.blue() / 255, lineNo);
		//viewer->visualizer()->setBackgroundColor(255, 255, 255);
		viewer->visualizer()->addLine(start, end, 255, 0, 0, lineNo);
        viewer->visualizer()->addText3D(std::to_string(i), middle, 0.025, 0, 0, 255, textNo);
        viewer->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, lineNo);
		viewer->visualizer()->removeAllPointClouds();
		
    }
}

void ToolWindowLineMatcher::showMatchedClouds()
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr srcCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::transformPointCloud(*m_colorCloudSrc, *srcCloud, m_pose);
    std::cout << "m_pose:" << std::endl << m_pose << std::endl;

    Eigen::Matrix3f rot = m_pose.topLeftCorner(3, 3);
    Eigen::Vector3f trans = m_pose.topRightCorner(3, 1);

    m_cloudViewer1->visualizer()->removeAllPointClouds();
    m_cloudViewer1->visualizer()->removeAllShapes();
    {
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> h1(srcCloud, 255, 0, 0);
        //m_cloudViewer1->visualizer()->addPointCloud(srcCloud, h1, "cloud1");
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> h2(m_colorCloudDst, 0, 0, 255);
        //m_cloudViewer1->visualizer()->addPointCloud(m_colorCloudDst, h2, "cloud2");
    }

    for (QMap<int, int>::iterator i = m_pairs.begin(); i != m_pairs.end(); i++)
    {
        LineSegment dstLine = m_flFrameDst.lines()->points[i.key()];
        LineSegment srcLine = m_flFrameSrc.lines()->points[i.value()];

        if (dstLine.index() == -1 || srcLine.index() == -1)
            continue;

        Eigen::Vector3f start = dstLine.start();
        Eigen::Vector3f end = dstLine.end();
        Eigen::Vector3f center = dstLine.center();
        QString lineNo = QString("dst_line_%1").arg(dstLine.index());
        QString textNo = QString("dst_id_%1").arg(dstLine.index());
        QString sphereNo = QString("dst_s_%1").arg(dstLine.index());
        pcl::PointXYZ ptStart, ptEnd, ptCenter;
        ptStart.getArray3fMap() = start;
        ptEnd.getArray3fMap() = end;
        ptCenter.getArray3fMap() = center;
        m_cloudViewer1->visualizer()->addLine(ptStart, ptEnd, 0, 0, 255, lineNo.toStdString());
        m_cloudViewer1->visualizer()->addSphere(ptCenter, 0.01, 0, 0, 255, sphereNo.toStdString());
        m_cloudViewer1->visualizer()->addText3D(std::to_string(dstLine.index()), ptCenter, 0.05, 0, 0, 255, textNo.toStdString());

        start = rot * srcLine.start() + trans;
        end = rot * srcLine.end() + trans;
        center = rot * srcLine.center() + trans;
        lineNo = QString("src_line_%1").arg(srcLine.index());
        textNo = QString("src_id_%1").arg(srcLine.index());
        sphereNo = QString("src_s_%1").arg(dstLine.index());
        ptStart.getArray3fMap() = start;
        ptEnd.getArray3fMap() = end;
        ptCenter.getArray3fMap() = center;
        m_cloudViewer1->visualizer()->addLine(ptStart, ptEnd, 255, 0, 0, lineNo.toStdString());
        m_cloudViewer1->visualizer()->addSphere(ptCenter, 0.01, 255, 0, 0, sphereNo.toStdString());
        m_cloudViewer1->visualizer()->addText3D(std::to_string(srcLine.index()), ptCenter, 0.05, 255, 0, 0, textNo.toStdString());
    }
}

void ToolWindowLineMatcher::stepCompute()
{
}

void ToolWindowLineMatcher::onActionLoadDataSet()
{
    m_device.reset(Device::createDevice());
    if (!m_device->open())
    {
        qDebug() << "Open device failed.";
        return;
    }

    int index1 = m_ui->comboBoxSrcFrame->currentIndex();
    int index2 = m_ui->comboBoxDstFrame->currentIndex();
    if (index1 < 0)
    {
        index1 = 0;
        index2 = 1;
    }
    m_ui->comboBoxSrcFrame->clear();
    m_ui->comboBoxDstFrame->clear();
    for (int i = 0; i < m_device->totalFrames(); i++)
    {
        m_ui->comboBoxSrcFrame->addItem(QString::number(i));
        m_ui->comboBoxDstFrame->addItem(QString::number(i));
    }
    m_ui->comboBoxSrcFrame->setCurrentIndex(index1);
    m_ui->comboBoxDstFrame->setCurrentIndex(index2);
    m_isLoaded = true;
    updateWidgets();
}

void ToolWindowLineMatcher::onActionMatch()
{
    compute();
    updateWidgets();
}

void ToolWindowLineMatcher::onActionMatchGpu()
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

void ToolWindowLineMatcher::onActionStep()
{
    QElapsedTimer timer;
    timer.start();
    m_lineMatcher->match(m_flFrameSrc.lines(), m_flFrameDst.lines(), m_tree, m_pairs, m_weights, m_pose);
    Eigen::Matrix4f pose = m_lineMatcher->step(m_flFrameSrc.lines(), m_flFrameDst.lines(), m_pose, m_error, m_pairs, m_weights);
    m_iterationDuration = timer.nsecsElapsed() / 1000000.f;
    m_totalDuration += m_iterationDuration;
    //m_flFrameSrc.setPose(pose);

    Eigen::Matrix3f rot = pose.topLeftCorner(3, 3);
    Eigen::Vector3f eulers = rot.eulerAngles(0, 1, 2);
    eulers.x() = qRadiansToDegrees(eulers.x());
    eulers.y() = qRadiansToDegrees(eulers.y());
    eulers.z() = qRadiansToDegrees(eulers.z());
    std::cout << "eulers: " << eulers.transpose() << std::endl;

    m_pose = pose * m_pose;

    std::cout << "pose:" << std::endl << pose << std::endl;
    std::cout << "m_pose:" << std::endl << m_pose << std::endl;
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
    if (index != m_ui->comboBoxSrcFrame->count() - 1)
        m_ui->comboBoxSrcFrame->setCurrentIndex(index + 1);
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
}

void ToolWindowLineMatcher::updateWidgets()
{
    m_ui->actionLoad_Data_Set->setEnabled(!m_isLoaded);
    m_ui->actionMatch_Gpu->setEnabled(m_isLoaded);
    m_ui->actionBegin_Step->setEnabled(m_isLoaded);
    m_ui->actionStep->setEnabled(m_isLoaded);

    m_ui->labelIteration->setText(QString::number(m_iteration));
    m_ui->labelError->setText(QString::number(m_error));
    m_ui->labelIterationDuration->setText(QString::number(m_iterationDuration));
    m_ui->labelTotalDuration->setText(QString::number(m_totalDuration));
}
