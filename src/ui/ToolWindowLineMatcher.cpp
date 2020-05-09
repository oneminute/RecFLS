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
    connect(m_ui->actionMatch_Gpu, &QAction::triggered, this, &ToolWindowLineMatcher::onActionMatchGpu);
    connect(m_ui->actionBegin_Step, &QAction::triggered, this, &ToolWindowLineMatcher::onActionBeginStep);
    connect(m_ui->actionStep, &QAction::triggered, this, &ToolWindowLineMatcher::onActionStep);
    connect(m_ui->actionStep_Rotation_Match, &QAction::triggered, this, &ToolWindowLineMatcher::onActionStepRotationMatch);
    connect(m_ui->actionStep_Translate_Match, &QAction::triggered, this, &ToolWindowLineMatcher::onActionStepTranslationMatch);
    connect(m_ui->actionReset, &QAction::triggered, this, &ToolWindowLineMatcher::onActionReset);

    connect(m_ui->comboBoxFirstFrame, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &ToolWindowLineMatcher::onComboBox1CurrentIndexChanged);
    connect(m_ui->pushButtonShowLineChainPair, &QPushButton::clicked, this, &ToolWindowLineMatcher::onActionShowPair);

    updateWidgets();
}

ToolWindowLineMatcher::~ToolWindowLineMatcher()
{
    if (m_isInit)
    {
        m_frameGpu1.free();
        m_frameGpu2.free();
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
        m_lineExtractor.reset(new LineExtractor);
    }

    if (!m_lineMatcher)
    {
        m_lineMatcher.reset(new LineMatcher);
    }

    m_lineMatcher->setMaxIterations(Settings::LineMatcher_MaxIterations.intValue());

    m_lineExtractor->setBoundaryCloudA1dThreshold(Settings::LineExtractor_BoundaryCloudA1dThreshold.value());
    m_lineExtractor->setCornerCloudA1dThreshold(Settings::LineExtractor_CornerCloudA1dThreshold.value());
    m_lineExtractor->setBoundaryCloudSearchRadius(Settings::LineExtractor_BoundaryCloudSearchRadius.value());
    m_lineExtractor->setCornerCloudSearchRadius(Settings::LineExtractor_CornerCloudSearchRadius.value());
    m_lineExtractor->setPCASearchRadius(Settings::LineExtractor_PCASearchRadius.value());
    m_lineExtractor->setMinNeighboursCount(Settings::LineExtractor_MinNeighboursCount.intValue());
    m_lineExtractor->setAngleCloudSearchRadius(Settings::LineExtractor_AngleCloudSearchRadius.value());
    m_lineExtractor->setAngleCloudMinNeighboursCount(Settings::LineExtractor_AngleCloudMinNeighboursCount.intValue());
    m_lineExtractor->setMinLineLength(Settings::LineExtractor_MinLineLength.value());
    m_lineExtractor->setBoundaryLineInterval(Settings::LineExtractor_BoundaryLineInterval.value());
    m_lineExtractor->setCornerLineInterval(Settings::LineExtractor_CornerLineInterval.value());
    m_lineExtractor->setBoundaryMaxZDistance(Settings::LineExtractor_BoundaryMaxZDistance.value());
    m_lineExtractor->setCornerMaxZDistance(Settings::LineExtractor_CornerMaxZDistance.value());
    m_lineExtractor->setBoundaryGroupLinesSearchRadius(Settings::LineExtractor_BoundaryGroupLinesSearchRadius.value());
    m_lineExtractor->setCornerGroupLinesSearchRadius(Settings::LineExtractor_CornerGroupLinesSearchRadius.value());

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

    if (!m_isInit)
    {
        m_frameGpu1.parameters.colorWidth = frame1.getColorWidth();
        m_frameGpu1.parameters.colorHeight = frame1.getColorHeight();
        m_frameGpu1.parameters.depthWidth = frame1.getDepthWidth();
        m_frameGpu1.parameters.depthHeight = frame1.getDepthHeight();
        m_frameGpu1.allocate();

        m_frameGpu2.parameters.colorWidth = frame2.getColorWidth();
        m_frameGpu2.parameters.colorHeight = frame2.getColorHeight();
        m_frameGpu2.parameters.depthWidth = frame2.getDepthWidth();
        m_frameGpu2.parameters.depthHeight = frame2.getDepthHeight();
        m_frameGpu2.allocate();
        m_isInit = true;
    }

    cv::cuda::GpuMat colorMatGpu1(frame1.getColorHeight(), frame1.getColorWidth(), CV_8UC3, m_frameGpu1.colorImage);
    cv::cuda::GpuMat depthMatGpu1(frame1.getDepthHeight(), frame1.getDepthWidth(), CV_16U, m_frameGpu1.depthImage);
    colorMatGpu1.upload(frame1.colorMat());
    depthMatGpu1.upload(frame1.depthMat());

    cv::cuda::GpuMat colorMatGpu2(frame2.getColorHeight(), frame2.getColorWidth(), CV_8UC3, m_frameGpu2.colorImage);
    cv::cuda::GpuMat depthMatGpu2(frame2.getDepthHeight(), frame2.getDepthWidth(), CV_16U, m_frameGpu2.depthImage);
    colorMatGpu2.upload(frame2.colorMat());
    depthMatGpu2.upload(frame2.depthMat());
    
    pcl::PointCloud<pcl::PointXYZI>::Ptr boundaryPoints1;
    pcl::PointCloud<pcl::PointXYZI>::Ptr boundaryPoints2;
    cv::Mat pointsMat1;
    cv::Mat pointsMat2;
    pcl::PointCloud<pcl::Normal>::Ptr normals1;
    pcl::PointCloud<pcl::Normal>::Ptr normals2;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cornerPoints1;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cornerPoints2;
    m_boundaryExtractor->setInputCloud(m_cloud1);
    m_boundaryExtractor->setWidth(frame1.getDepthWidth());
    m_boundaryExtractor->setHeight(frame1.getDepthHeight());
    m_boundaryExtractor->setCx(frame1.getDevice()->cx());
    m_boundaryExtractor->setCy(frame1.getDevice()->cy());
    m_boundaryExtractor->setFx(frame1.getDevice()->fx());
    m_boundaryExtractor->setFy(frame1.getDevice()->fy());
    m_boundaryExtractor->setNormals(nullptr);
    m_boundaryExtractor->computeCUDA(m_frameGpu1);
    boundaryPoints1 = m_boundaryExtractor->boundaryPoints();
    m_filteredCloud1 = m_boundaryExtractor->filteredCloud();
    pointsMat1 = m_boundaryExtractor->pointsMat();
    normals1 = m_boundaryExtractor->normals();
    cornerPoints1 = m_boundaryExtractor->cornerPoints();

    m_boundaryExtractor->setInputCloud(m_cloud2);
    m_boundaryExtractor->setWidth(frame2.getDepthWidth());
    m_boundaryExtractor->setHeight(frame2.getDepthHeight());
    m_boundaryExtractor->setCx(frame2.getDevice()->cx());
    m_boundaryExtractor->setCy(frame2.getDevice()->cy());
    m_boundaryExtractor->setFx(frame2.getDevice()->fx());
    m_boundaryExtractor->setFy(frame2.getDevice()->fy());
    m_boundaryExtractor->setNormals(nullptr);
    m_boundaryExtractor->computeCUDA(m_frameGpu2);
    boundaryPoints2 = m_boundaryExtractor->boundaryPoints();
    m_filteredCloud2 = m_boundaryExtractor->filteredCloud();
    pointsMat2 = m_boundaryExtractor->pointsMat();
    normals2 = m_boundaryExtractor->normals();
    cornerPoints2 = m_boundaryExtractor->cornerPoints();

    QList<LineSegment> lines1;
    QList<LineSegment> lines2;
    Eigen::Vector3f center1;
    Eigen::Vector3f center2;
    lines1 = m_lineExtractor->compute(boundaryPoints1, cornerPoints1);
    qDebug() << pointsMat1.type();
    m_lineCloud1 = m_lineExtractor->lineCloud();

    lines2 = m_lineExtractor->compute(boundaryPoints2, cornerPoints1);
    m_lineCloud2 = m_lineExtractor->lineCloud();

    m_cloudViewer1->visualizer()->removeAllPointClouds();
    m_cloudViewer1->visualizer()->removeAllShapes();
    m_cloudViewer2->visualizer()->removeAllPointClouds();
    m_cloudViewer2->visualizer()->removeAllShapes();
    m_cloudViewer3->visualizer()->removeAllPointClouds();
    m_cloudViewer3->visualizer()->removeAllShapes();

    m_tree.reset(new pcl::KdTreeFLANN<Line>());
    m_tree->setInputCloud(m_lineCloud2);
    //qDebug() << "msl point cloud2:" << m_mslPointCloud2->size();

    // œ‘ æµ„‘∆
    {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> h1(m_colorCloud1, 127, 127, 127);
        m_cloudViewer2->visualizer()->addPointCloud(m_colorCloud1, h1, "cloud");
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> h2(m_colorCloud2, 127, 127, 127);
        m_cloudViewer3->visualizer()->addPointCloud(m_colorCloud2, h2, "cloud");
    }
    showCloudAndLines(m_cloudViewer2, lines1, m_lineCloud1);
    showCloudAndLines(m_cloudViewer3, lines2, m_lineCloud2);

    m_rotationDelta = Eigen::Matrix3f::Identity();
    m_translationDelta = Eigen::Vector3f::Zero();
    m_rotation = Eigen::Matrix3f::Identity();
    m_translation = Eigen::Vector3f::Zero();
    m_m = Eigen::Matrix4f::Identity();

    m_isInit = true;
}

void ToolWindowLineMatcher::compute()
{
    initCompute();
    m_m = m_lineMatcher->compute(m_lineCloud1, m_lineCloud2, m_rotationError, m_translationError);

    m_ui->comboBoxLineChainPairs->clear();
    
    showMatchedClouds();
    updateWidgets();
}

void ToolWindowLineMatcher::showCloudAndLines(CloudViewer* viewer, QList<LineSegment>& lines, boost::shared_ptr<pcl::PointCloud<Line>>& mslCloud)
{
    QColor color;

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
        Line msl = mslCloud->points[i];
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
    //Eigen::Matrix4f rotMat(Eigen::Matrix4f::Identity());
    //rotMat.topLeftCorner(3, 3) = m_rotation;
    //rotMat.topRightCorner(3, 1) = m_translation;
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr m_tmpCloud1(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::transformPointCloud(*m_colorCloud1, *m_tmpCloud1, m_m);

    {
        m_cloudViewer1->visualizer()->removeAllPointClouds();
        m_cloudViewer1->visualizer()->removeAllShapes();

        //pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> h1(m_tmpCloud1);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> h1(m_tmpCloud1, 255, 0, 0);
        m_cloudViewer1->visualizer()->addPointCloud(m_tmpCloud1, h1, "cloud1");
        //pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> h2(m_colorCloud2);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> h2(m_colorCloud2, 0, 0, 255);
        m_cloudViewer1->visualizer()->addPointCloud(m_colorCloud2, h2, "cloud2");

    }

    for (int i = 0; i < m_lineCloud1->size(); i++)
    {
        Line line = m_lineCloud1->points[i];
        pcl::PointXYZI start, end, middle;
        start.getVector3fMap() = line.getEndPoint(-3);
        end.getVector3fMap() = line.getEndPoint(3);
        middle.getVector3fMap() = line.point;
        QString lineName = QString("lines1_%1").arg(i);
        std::string textNo = "lines1_text_" + std::to_string(i);
        m_cloudViewer1->visualizer()->addText3D(std::to_string(i), middle, 0.05, 255, 0, 0, textNo);
        m_cloudViewer1->visualizer()->addLine(start, end, 255, 0, 0, lineName.toStdString());
        m_cloudViewer1->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, lineName.toStdString());
    }

    for (int i = 0; i < m_lineCloud2->size(); i++)
    {
        Line line = m_lineCloud2->points[i];
        pcl::PointXYZI start, end, middle;
        start.getVector3fMap() = line.getEndPoint(-3);
        end.getVector3fMap() = line.getEndPoint(3);
        middle.getVector3fMap() = line.point;
        QString lineName = QString("lines2_%1").arg(i);
        std::string textNo = "lines2_text_" + std::to_string(i);
        m_cloudViewer1->visualizer()->addText3D(std::to_string(i), middle, 0.05, 0, 0, 255, textNo);
        m_cloudViewer1->visualizer()->addLine(start, end, 0, 0, 255, lineName.toStdString());
        m_cloudViewer1->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, lineName.toStdString());
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
    Eigen::Matrix4f M = m_lineMatcher->step(m_lineCloud1, m_lineCloud2, m_tree, m_rotationError, m_translationError, m_pairs);
    m_m = M * m_m;
    m_iteration++;
    showMatchedClouds();
    updateWidgets();
}

void ToolWindowLineMatcher::onActionStepRotationMatch()
{
    m_rotationDelta = m_lineMatcher->stepRotation(m_lineCloud1, m_lineCloud2, m_tree, m_pairs);
    m_translationDelta = Eigen::Vector3f::Zero();

    m_rotation = m_rotation * m_rotationDelta;

    m_iteration++;

    showMatchedClouds();
    //updateWidgets();
}

void ToolWindowLineMatcher::onActionStepTranslationMatch()
{
    m_translationDelta = m_lineMatcher->stepTranslation(m_lineCloud1, m_lineCloud2, m_tree, m_pairs);
    m_rotationDelta = Eigen::Quaternionf::Identity();

    m_translation += m_translationDelta;

    m_iteration++;

    showMatchedClouds();
    //updateWidgets();
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
    m_ui->actionMatch->setEnabled(!m_isInit && m_isLoaded);
    m_ui->actionBegin_Step->setEnabled(!m_isInit && m_isLoaded);
    m_ui->actionStep_Rotation_Match->setEnabled(m_isInit && m_isStepMode);
    m_ui->actionStep_Translate_Match->setEnabled(m_isInit && m_isStepMode);
    m_ui->actionReset->setEnabled(m_isLoaded);

    m_ui->labelIteration->setText(QString::number(m_iteration));
    m_ui->labelRotationError->setText(QString::number(qRadiansToDegrees(m_rotationError)));
    m_ui->labelTranslationError->setText(QString::number(m_translationError));
}
