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
    //, m_diameter1(0)
    //, m_diameter2(0)
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
    if (!m_boundaryExtractor)
    {
        m_boundaryExtractor.reset(new BoundaryExtractor);
    }

    if (!m_lineExtractor)
    {
        m_lineExtractor.reset(new FusedLineExtractor);
    }

    if (!m_lineMatcher)
    {
        m_lineMatcher.reset(new LineMatcher);
    }

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

    cv::cuda::GpuMat colorMatGpuSrc(frameSrc.getColorHeight(), frameSrc.getColorWidth(), CV_8UC3, m_frameGpuBESrc.colorImage);
    cv::cuda::GpuMat depthMatGpuSrc(frameSrc.getDepthHeight(), frameSrc.getDepthWidth(), CV_16U, m_frameGpuBESrc.depthImage);
    colorMatGpuSrc.upload(frameSrc.colorMat());
    depthMatGpuSrc.upload(frameSrc.depthMat());

    cv::cuda::GpuMat colorMatGpuDst(frameDst.getColorHeight(), frameDst.getColorWidth(), CV_8UC3, m_frameGpuBEDst.colorImage);
    cv::cuda::GpuMat depthMatGpuDst(frameDst.getDepthHeight(), frameDst.getDepthWidth(), CV_16U, m_frameGpuBEDst.depthImage);
    colorMatGpuDst.upload(frameDst.colorMat());
    depthMatGpuDst.upload(frameDst.depthMat());
    
    m_lineExtractor->compute(frameSrc, m_frameGpuBESrc);
    //m_linesSrc = m_lineExtractor->lines();
    m_linesCloudSrc = m_lineExtractor->linesCloud();
    m_beCloudSrc = m_lineExtractor->allBoundary();
    m_groupPointsSrc = m_lineExtractor->groupPoints();

    m_lineExtractor->compute(frameDst, m_frameGpuBEDst);
    //m_linesDst = m_lineExtractor->lines();
    m_linesCloudDst = m_lineExtractor->linesCloud();
    m_beCloudDst = m_lineExtractor->allBoundary();
    m_groupPointsDst = m_lineExtractor->groupPoints();

    m_cloudViewer1->visualizer()->removeAllPointClouds();
    m_cloudViewer1->visualizer()->removeAllShapes();
    m_cloudViewer2->visualizer()->removeAllPointClouds();
    m_cloudViewer2->visualizer()->removeAllShapes();
    m_cloudViewer3->visualizer()->removeAllPointClouds();
    m_cloudViewer3->visualizer()->removeAllShapes();

    m_tree.reset(new pcl::KdTreeFLANN<LineSegment>());
    m_tree->setInputCloud(m_linesCloudDst);
    qDebug() << "msl point cloud2:" << m_linesCloudDst->size();

    Eigen::AngleAxisf rollAngle(M_PI / 72, Eigen::Vector3f::UnitZ());
    Eigen::AngleAxisf yawAngle(0, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf pitchAngle(0, Eigen::Vector3f::UnitX());
    Eigen::Quaternion<float> q = rollAngle * yawAngle * pitchAngle;
    Eigen::Matrix3f rotationMatrix = q.matrix();

    Eigen::Matrix3f rotation = Eigen::Matrix3f::Identity();
    Eigen::Vector3f translation = Eigen::Vector3f::Zero();
    //Eigen::Matrix3f rotation = rotationMatrix;
    //Eigen::Vector3f translation = Eigen::Vector3f(0.05f, 0.02f, 0);
    m_pose = Eigen::Matrix4f::Identity();
    m_pose.topLeftCorner(3, 3) = rotation;
    m_pose.topRightCorner(3, 1) = translation;

    m_iteration = 0;

    m_lineMatcher->match(m_linesCloudSrc, m_linesCloudDst, m_tree, rotation, translation, m_pairs);

    // œ‘ æµ„‘∆
    {
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> behSrc(m_beCloudSrc, "intensity");
        m_cloudViewer2->visualizer()->addPointCloud(m_beCloudSrc, behSrc, "cloud_src");
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> behDst(m_beCloudDst, "intensity");
        m_cloudViewer3->visualizer()->addPointCloud(m_beCloudDst, behDst, "cloud_dst");
    }

    showCloudAndLines(m_cloudViewer2, m_linesCloudSrc);
    showCloudAndLines(m_cloudViewer3, m_linesCloudDst);
    showMatchedClouds();

    m_isInit = true;
}

void ToolWindowLineMatcher::compute()
{
    initCompute();
    Eigen::Matrix3f rot = m_pose.topLeftCorner(3, 3);
    Eigen::Vector3f trans = m_pose.topRightCorner(3, 1);
    m_pose = m_lineMatcher->compute(m_linesCloudSrc, m_linesCloudDst, rot, trans, m_rotationError, m_translationError);

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
        viewer->visualizer()->addLine(start, end, line.red() / 255, line.green() / 255, line.blue() / 255, lineNo);
        viewer->visualizer()->addText3D(std::to_string(i), middle, 0.025, 1, 1, 1, textNo);
        viewer->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, lineNo);
    }
}

void ToolWindowLineMatcher::showMatchedClouds()
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr m_tmpCloud1(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::transformPointCloud(*m_colorCloudSrc, *m_tmpCloud1, m_pose);

    m_cloudViewer1->visualizer()->removeAllPointClouds();
    m_cloudViewer1->visualizer()->removeAllShapes();
    {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> h1(m_tmpCloud1, 255, 0, 0);
        m_cloudViewer1->visualizer()->addPointCloud(m_tmpCloud1, h1, "cloud1");
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> h2(m_colorCloudDst, 0, 0, 255);
        m_cloudViewer1->visualizer()->addPointCloud(m_colorCloudDst, h2, "cloud2");

    }
    Eigen::Matrix3f rotation = m_pose.topLeftCorner(3, 3);
    Eigen::Vector3f translation = m_pose.topRightCorner(3, 1);

    for (QMap<int, int>::iterator i = m_pairs.begin(); i != m_pairs.end(); i++)
    {
        LineSegment dstLine = m_linesCloudDst->points[i.key()];
        LineSegment srcLine = m_linesCloudSrc->points[i.value()];

        if (dstLine.index() == -1 || srcLine.index() == -1)
            continue;

        Eigen::Vector3f start = dstLine.start();
        Eigen::Vector3f end = dstLine.end();
        Eigen::Vector3f middle = dstLine.middle();
        QString lineNo = QString("dst_line_%1").arg(dstLine.index());
        QString textNo = QString("dst_id_%1").arg(dstLine.index());
        pcl::PointXYZ ptStart, ptEnd, ptMiddle;
        ptStart.getArray3fMap() = start;
        ptEnd.getArray3fMap() = end;
        ptMiddle.getArray3fMap() = middle;
        m_cloudViewer1->visualizer()->addLine(ptStart, ptEnd, 0, 255, 0, lineNo.toStdString());
        m_cloudViewer1->visualizer()->addText3D(std::to_string(dstLine.index()), ptMiddle, 0.025, 0, 255, 0, textNo.toStdString());

        start = rotation * srcLine.start() + translation;
        end = rotation * srcLine.end() + translation;
        middle = rotation * srcLine.middle() + translation;
        lineNo = QString("src_line_%1").arg(srcLine.index());
        textNo = QString("src_id_%1").arg(srcLine.index());
        ptStart.getArray3fMap() = start;
        ptEnd.getArray3fMap() = end;
        ptMiddle.getArray3fMap() = middle;
        m_cloudViewer1->visualizer()->addLine(ptStart, ptEnd, 255, 255, 255, lineNo.toStdString());
        m_cloudViewer1->visualizer()->addText3D(std::to_string(srcLine.index()), ptMiddle, 0.025, 1, 1, 1, textNo.toStdString());
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
    Eigen::Matrix3f rot = m_pose.topLeftCorner(3, 3);
    Eigen::Vector3f trans = m_pose.topRightCorner(3, 1);
    Eigen::Matrix4f pose = m_lineMatcher->step(m_linesCloudSrc, m_linesCloudDst, m_tree, rot, trans, m_rotationError, m_translationError, m_pairs);
    rot = pose.topLeftCorner(3, 3);
    Eigen::Vector3f eulers = rot.eulerAngles(0, 1, 2);
    eulers.x() = qRadiansToDegrees(eulers.x());
    eulers.y() = qRadiansToDegrees(eulers.y());
    eulers.z() = qRadiansToDegrees(eulers.z());
    std::cout << eulers.transpose() << std::endl;
    m_pose = pose * m_pose;
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

    int index2 = m_pairIndices[m_ui->comboBoxLineChainPairs->currentIndex()];
    int index1 = m_pairs[index2];
    //LineChain lc1 = m_chains1[index1];
    //LineChain lc2 = m_chains2[index2];
    //qDebug() << index1 << lc1.name() << index2 << lc2.name();

    //{
    //    pcl::PointXYZ start, end;
    //    //MSL msl = m_mslCloud1->points[lc1.line1];
    //    start.getVector3fMap() = lc1.line1.getEndPoint(-3);
    //    end.getVector3fMap() = lc1.line1.getEndPoint(3);
    //    m_cloudViewer2->visualizer()->addLine(start, end, 255, 0, 0, "chain_line_1");
    //    m_cloudViewer2->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "chain_line_1");

    //    //msl = m_mslCloud1->points[lc1.line2];
    //    start.getVector3fMap() = lc1.line2.getEndPoint(-3);
    //    end.getVector3fMap() = lc1.line2.getEndPoint(3);
    //    m_cloudViewer2->visualizer()->addLine(start, end, 0, 0, 255, "chain_line_2");
    //    m_cloudViewer2->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "chain_line_2");

    //    start.getVector3fMap() = lc1.point1;
    //    end.getVector3fMap() = lc1.point2;
    //    m_cloudViewer2->visualizer()->addLine(start, end, 0, 255, 0, "chain_line");
    //    m_cloudViewer2->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "chain_line");

    //    end.getVector3fMap() = lc1.point;
    //    start.getVector3fMap() = lc1.point + lc1.xLocal * 0.2f;
    //    m_cloudViewer2->visualizer()->addArrow(start, end, 255, 0, 0, false, "xaxis");
    //    start.getVector3fMap() = lc1.point + lc1.yLocal * 0.2f;
    //    m_cloudViewer2->visualizer()->addArrow(start, end, 0, 255, 0, false, "yaxis");
    //    start.getVector3fMap() = lc1.point + lc1.zLocal * 0.2f;
    //    m_cloudViewer2->visualizer()->addArrow(start, end, 0, 0, 255, false, "zaxis");
    //}

    //{
    //    pcl::PointXYZ start, end;
    //    //MSL msl = m_mslCloud2->points[lc2.line1];
    //    start.getVector3fMap() = lc2.line1.getEndPoint(-3);
    //    end.getVector3fMap() = lc2.line1.getEndPoint(3);
    //    m_cloudViewer3->visualizer()->addLine(start, end, 255, 0, 0, "chain_line_1");
    //    m_cloudViewer3->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "chain_line_1");

    //    //msl = m_mslCloud2->points[lc2.line2];
    //    start.getVector3fMap() = lc2.line2.getEndPoint(-3);
    //    end.getVector3fMap() = lc2.line2.getEndPoint(3);
    //    m_cloudViewer3->visualizer()->addLine(start, end, 0, 0, 255, "chain_line_2");
    //    m_cloudViewer3->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "chain_line_2");

    //    start.getVector3fMap() = lc2.point1;
    //    end.getVector3fMap() = lc2.point2;
    //    m_cloudViewer3->visualizer()->addLine(start, end, 0, 255, 0, "chain_line");
    //    m_cloudViewer3->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "chain_line");

    //    end.getVector3fMap() = lc2.point;
    //    start.getVector3fMap() = lc2.point + lc2.xLocal * 0.2f;
    //    m_cloudViewer3->visualizer()->addArrow(start, end, 255, 0, 0, false, "xaxis");
    //    start.getVector3fMap() = lc2.point + lc2.yLocal * 0.2f;
    //    m_cloudViewer3->visualizer()->addArrow(start, end, 0, 255, 0, false, "yaxis");
    //    start.getVector3fMap() = lc2.point + lc2.zLocal * 0.2f;
    //    m_cloudViewer3->visualizer()->addArrow(start, end, 0, 0, 255, false, "zaxis");
    //}
}

void ToolWindowLineMatcher::updateWidgets()
{
    m_ui->actionLoad_Data_Set->setEnabled(!m_isLoaded);
    m_ui->actionMatch->setEnabled(m_isLoaded);
    m_ui->actionBegin_Step->setEnabled(m_isLoaded);
    m_ui->actionStep->setEnabled(m_isLoaded);
    //m_ui->actionStep_Rotation_Match->setEnabled(m_isInit && m_isStepMode);
    //m_ui->actionStep_Translate_Match->setEnabled(m_isInit && m_isStepMode);
    //m_ui->actionReset->setEnabled(m_isLoaded);

    m_ui->labelIteration->setText(QString::number(m_iteration));
    m_ui->labelRotationError->setText(QString::number(qRadiansToDegrees(m_rotationError)));
    m_ui->labelTranslationError->setText(QString::number(m_translationError));
}
