#include "ToolWindowICPMatcher.h"
#include "ui_ToolWindowICPMatcher.h"

#include <QDebug>
#include <QtMath>
#include <QPushButton>
#include <QFile>
#include <QTextStream>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/common/pca.h>

#include "device/SensorReaderDevice.h"
#include "matcher/ICPMatcher.h"
#include "common/Parameters.h"
#include "util/Utils.h"
#include "util/StopWatch.h"
#include "cuda/cuda.hpp"


ToolWindowICPMatcher::ToolWindowICPMatcher(QWidget *parent) 
    : QMainWindow(parent)
    , m_ui(new Ui::ToolWindowICPMatcher)
    , m_isStepMode(false)
    , m_isInit(false)
    , m_isLoaded(false)
    , m_iteration(0)
    , m_rotationDelta(Eigen::Quaternionf::Identity())
    , m_translationDelta(Eigen::Vector3f::Zero())
    , m_rotationError(0)
    , m_translationError(0)
{
    m_ui->setupUi(this);

    m_cloudViewer = new CloudViewer;

    m_cloudViewer->setCameraPosition(0, 0, -1.5f, 0, 0, 0, 0, -1, 0);

    m_ui->verticalLayoutCloud->addWidget(m_cloudViewer);

    connect(m_ui->actionLoad_Data_Set, &QAction::triggered, this, &ToolWindowICPMatcher::onActionLoadDataSet);
    connect(m_ui->actionMatch, &QAction::triggered, this, &ToolWindowICPMatcher::onActionMatch);
    connect(m_ui->actionCompute_GPU, &QAction::triggered, this, &ToolWindowICPMatcher::onActionComputeGPU);
    connect(m_ui->actionStep_Reset, &QAction::triggered, this, &ToolWindowICPMatcher::onActionStepReset);
    connect(m_ui->actionStep, &QAction::triggered, this, &ToolWindowICPMatcher::onActionStep);
    connect(m_ui->actionStep_GPU, &QAction::triggered, this, &ToolWindowICPMatcher::onActionStepGPU);

    connect(m_ui->comboBoxFrameSrc, QOverload<int>::of(&QComboBox::currentIndexChanged), this, &ToolWindowICPMatcher::onComboBoxFrameSrcCurrentIndexChanged);

    updateWidgets();
}

ToolWindowICPMatcher::~ToolWindowICPMatcher()
{
    if (m_isInit)
    {
        m_frameSrc.free();
        m_frameDst.free();
    }
}

void ToolWindowICPMatcher::initCompute()
{
    m_icp.reset(new ICPMatcher);

    int frameIndexSrc = m_ui->comboBoxFrameSrc->currentIndex();
    Frame frameSrc = m_device->getFrame(frameIndexSrc);
    int frameIndexDst = m_ui->comboBoxFrameDst->currentIndex();
    //int frameIndexDst = m_ui->comboBoxFrameSrc->currentIndex();
    Frame frameDst = m_device->getFrame(frameIndexDst);

    m_ui->widgetImageSrc->setImage(cvMat2QImage(frameSrc.colorMat()));
    m_ui->widgetImageDst->setImage(cvMat2QImage(frameDst.colorMat()));

    if (!m_isInit)
    {
        cuda::IcpParameters parameters;
        parameters.cx = m_device->cx();
        parameters.cy = m_device->cy();
        parameters.fx = m_device->fx();
        parameters.fy = m_device->fy();
        parameters.minDepth = Settings::BoundaryExtractor_MinDepth.value();
        parameters.maxDepth = Settings::BoundaryExtractor_MaxDepth.value();
        parameters.depthShift = m_device->depthShift();
        parameters.normalKernalRadius = Settings::ICPMatcher_CudaNormalKernalRadius.intValue();
        parameters.normalKnnRadius = Settings::ICPMatcher_CudaNormalKnnRadius.value();
        parameters.depthWidth = frameSrc.getDepthWidth();
        parameters.depthHeight = frameSrc.getDepthHeight();
        parameters.icpAnglesThreshold = Settings::ICPMatcher_AnglesThreshold.value();
        parameters.icpDistThreshold = Settings::ICPMatcher_DistanceThreshold.value();
        parameters.icpKernalRadius = Settings::ICPMatcher_IcpKernelRadius.intValue();
        parameters.blockSize = Settings::ICPMatcher_CudaBlockSize.intValue();

        m_frameSrc.parameters = parameters;
        m_frameDst.parameters = parameters;
        m_cache.parameters = parameters;

        m_frameSrc.allocate();
        m_frameDst.allocate();
        m_cache.allocate();

        m_isInit = true;
    }

    cv::cuda::GpuMat depthMatGpu1(frameSrc.getDepthHeight(), frameSrc.getDepthWidth(), CV_16U, m_frameSrc.depthImage);
    depthMatGpu1.upload(frameDst.depthMat());

    cv::cuda::GpuMat depthMatGpu2(frameDst.getDepthHeight(), frameDst.getDepthWidth(), CV_16U, m_frameDst.depthImage);
    depthMatGpu2.upload(frameSrc.depthMat());

    TICK("generate_cloud");
    cuda::icpGenerateCloud(m_frameSrc);
    cuda::icpGenerateCloud(m_frameDst);
    m_cache.srcCloud = m_frameSrc.pointCloud;
    m_cache.dstCloud = m_frameDst.pointCloud;
    m_cache.srcNormals = m_frameSrc.pointCloudNormals;
    m_cache.dstNormals = m_frameDst.pointCloudNormals;
    TOCK("generate_cloud");

    TICK("icp_cloud_downloading");
    std::vector<float3> pointsSrc;
    m_frameSrc.pointCloud.download(pointsSrc);
    std::vector<float3> normalsSrc;
    m_frameSrc.pointCloudNormals.download(normalsSrc);
    std::vector<float3> pointsDst;
    m_frameDst.pointCloud.download(pointsDst);
    std::vector<float3> normalsDst;
    m_frameDst.pointCloudNormals.download(normalsDst);

    m_cloudSrc.reset(new pcl::PointCloud<pcl::PointXYZI>);
    m_cloudDst.reset(new pcl::PointCloud<pcl::PointXYZI>);
    m_normalsSrc.reset(new pcl::PointCloud<pcl::Normal>);
    m_normalsDst.reset(new pcl::PointCloud<pcl::Normal>);
    for(int i = 0; i < frameSrc.getDepthHeight(); i++) 
    {
        for(int j = 0; j < frameSrc.getDepthWidth(); j++) 
        {
            int index = i * frameSrc.getDepthWidth() + j;

            Eigen::Vector3f valueSrc = toVector3f(pointsSrc[index]);
            Eigen::Vector3f valueDst = toVector3f(pointsDst[index]);
            pcl::PointXYZI ptSrc;
            pcl::Normal normalSrc;
            pcl::PointXYZI ptDst;
            pcl::Normal normalDst;

            if (!(isnan(valueSrc.x()) || isnan(valueSrc.y()) || isnan(valueSrc.z())))
            {
                ptSrc.getVector3fMap() = valueSrc;
                ptSrc.intensity = index;
                normalSrc.getNormalVector3fMap() = toVector3f(normalsSrc[index]);

                m_cloudSrc->points.push_back(ptSrc);
                m_normalsSrc->points.push_back(normalSrc);
            }

            if (!(isnan(valueDst.x()) || isnan(valueDst.y()) || isnan(valueDst.z())))
            {
                ptDst.getVector3fMap() = valueDst;
                ptDst.intensity = index;
                normalDst.getNormalVector3fMap() = toVector3f(normalsDst[index]);

                m_cloudDst->points.push_back(ptDst);
                m_normalsDst->points.push_back(normalDst);
            }
        }
    }

    m_cloudSrc->width = m_cloudSrc->points.size();
    m_cloudSrc->height = 1;
    m_cloudSrc->is_dense = true;

    m_cloudDst->width = m_cloudDst->points.size();
    m_cloudDst->height = 1;
    m_cloudDst->is_dense = true;

    m_normalsSrc->width = m_normalsSrc->points.size();
    m_normalsSrc->height = 1;
    m_normalsSrc->is_dense = true;

    m_normalsDst->width = m_normalsDst->points.size();
    m_normalsDst->height = 1;
    m_normalsDst->is_dense = true;
    TOCK("icp_cloud_downloading");

    m_tree.reset(new pcl::search::KdTree<pcl::PointXYZI>);
    m_tree->setInputCloud(m_cloudDst);

    m_rotationDelta = Eigen::Matrix3f::Identity();
    m_translationDelta = Eigen::Vector3f::Zero();

    Eigen::AngleAxisf rollAngle(M_PI / 72, Eigen::Vector3f::UnitZ());
    Eigen::AngleAxisf yawAngle(0, Eigen::Vector3f::UnitY());
    Eigen::AngleAxisf pitchAngle(0, Eigen::Vector3f::UnitX());
    Eigen::Quaternion<float> q = rollAngle * yawAngle * pitchAngle;
    Eigen::Matrix3f rotationMatrix = q.matrix();

    m_rotation = Eigen::Matrix3f::Identity();
    m_translation = Eigen::Vector3f::Zero();
    //m_rotation = rotationMatrix;
    //m_translation = Eigen::Vector3f(0.05f, 0.02f, 0);
    m_pose = Eigen::Matrix4f::Identity();
    m_pose.topLeftCorner(3, 3) = m_rotation;
    m_pose.topRightCorner(3, 1) = m_translation;
    m_iteration = 0;

    // ÏÔÊ¾µãÔÆ
    //{
    //    pcl::transformPointCloud(*m_cloudSrc, *m_cloudSrc, m_pose);
    //    m_cloudViewer->removeAllClouds();
    //    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> h1(m_cloudSrc, 255, 0, 0);
    //    m_cloudViewer->visualizer()->addPointCloud(m_cloudSrc, h1, "cloud_src");
    //    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> h2(m_cloudDst, 0, 0, 255);
    //    m_cloudViewer->visualizer()->addPointCloud(m_cloudDst, h2, "cloud_dst");
    //    //m_cloudViewer->visualizer()->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(m_cloudSrc, m_normalsSrc, 100, 0.1f);
    //    //m_cloudViewer->visualizer()->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(m_cloudDst, m_normalsDst, 100, 0.1f);
    //}
    showMatchedClouds();

    m_isInit = true;
}

void ToolWindowICPMatcher::compute()
{
    initCompute();

    TICK("icp");
    float error = 0;
    m_icp->stepGPU(m_cache, m_rotation, m_translation, error);
    TOCK("icp");

    //showMatchedClouds();
    //updateWidgets();
    m_isStepMode = false;

    StopWatch::instance().debugPrint();
}

void ToolWindowICPMatcher::showMatchedClouds()
{
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_tmpCloud1(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::transformPointCloud(*m_cloudSrc, *m_tmpCloud1, m_pose);

    {
        m_cloudViewer->visualizer()->removeAllPointClouds();
        m_cloudViewer->visualizer()->removeAllShapes();

        //pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> h1(m_tmpCloud1);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> h1(m_tmpCloud1, 255, 0, 0);
        m_cloudViewer->visualizer()->addPointCloud(m_tmpCloud1, h1, "cloud1");
        //pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> h2(m_colorCloud2);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> h2(m_cloudDst, 0, 0, 255);
        m_cloudViewer->visualizer()->addPointCloud(m_cloudDst, h2, "cloud2");

    }
}

void ToolWindowICPMatcher::stepCompute()
{
}

void ToolWindowICPMatcher::onActionLoadDataSet()
{
    m_device.reset(new SensorReaderDevice);
    if (!m_device->open())
    {
        qDebug() << "Open device failed.";
        return;
    }

    int index1 = m_ui->comboBoxFrameSrc->currentIndex();
    int index2 = m_ui->comboBoxFrameDst->currentIndex();
    if (index1 < 0)
    {
        index1 = 0;
        index2 = 1;
    }
    m_ui->comboBoxFrameSrc->clear();
    m_ui->comboBoxFrameDst->clear();
    for (int i = 0; i < m_device->totalFrames(); i++)
    {
        m_ui->comboBoxFrameSrc->addItem(QString::number(i));
        m_ui->comboBoxFrameDst->addItem(QString::number(i));
    }
    m_ui->comboBoxFrameSrc->setCurrentIndex(index1);
    m_ui->comboBoxFrameDst->setCurrentIndex(index2);
    m_isLoaded = true;
    updateWidgets();
}

void ToolWindowICPMatcher::onActionMatch()
{
    compute();
    updateWidgets();
}

void ToolWindowICPMatcher::onActionComputeGPU()
{
    initCompute();

    float error = 0;
    m_pose = m_icp->compute(m_cache, m_rotation, m_translation, error);
    m_iteration = Settings::ICPMatcher_MaxIterations.intValue();

    showMatchedClouds();
    updateWidgets();
    StopWatch::instance().debugPrint();
}

void ToolWindowICPMatcher::onActionStepReset()
{
    initCompute();

    m_isStepMode = true;
    updateWidgets();
}

void ToolWindowICPMatcher::onActionStep()
{
    Eigen::Matrix4f poseDelta = m_icp->step(m_cloudSrc, m_cloudDst, m_normalsSrc, m_normalsDst, m_tree, m_rotation, m_translation, 0.05f, 0.95f, m_pairs, m_error);
    m_pose = poseDelta * m_pose;
    m_rotation = m_pose.topLeftCorner(3, 3);
    m_translation = m_pose.topRightCorner(3, 1);
    m_iteration++;

    showMatchedClouds();
    updateWidgets();
    StopWatch::instance().debugPrint();
}

void ToolWindowICPMatcher::onActionStepGPU()
{
    TICK("icp");
    float error = 0;
    Eigen::Matrix4f poseDelta = m_icp->stepGPU(m_cache, m_rotation, m_translation, error);
    TOCK("icp");

    m_pose = poseDelta * m_pose;
    m_rotation = m_pose.topLeftCorner(3, 3);
    m_translation = m_pose.topRightCorner(3, 1);
    m_iteration++;
    std::vector<int> pairs;
    m_cache.pairs.download(pairs);

    m_pairs = 0;
    std::cout << "out result" << std::endl;
    QFile file("gpu_pairs.txt");
    file.open(QIODevice::WriteOnly | QFile::Truncate | QFile::Text);
    QTextStream out(&file);
    //QString strBuffer;
    for (int i = 0; i < pairs.size(); i++)
    {
        if (pairs[i] < 0)
            continue;

        int x1, y1, x2, y2;
        x1 = i % 640;
        y1 = i / 640;
        x2 = pairs[i] % 640;
        y2 = pairs[i] / 640;
        out << "[" << x1 << ", " << y1 << "] -- [" << x2 << ", " << y2 << "] " << (x1 - x2) << ", " << (y1 - y2) << "\n";
        //strBuffer.append("[%1, %2] -- [%3, %4]\r\n").arg(x1).arg(y1).arg(x2).arg(y2);

        m_pairs++;
    }

    //out << strBuffer;
    file.close();
    std::cout << "out complete" << std::endl;

    showMatchedClouds();
    updateWidgets();
    StopWatch::instance().debugPrint();
}

void ToolWindowICPMatcher::onActionReset()
{
    m_isInit = false;
    m_isStepMode = false;
    m_isLoaded = false;
    m_iteration = 0;

    updateWidgets();
}

void ToolWindowICPMatcher::onComboBoxFrameSrcCurrentIndexChanged(int index)
{
    if (index != m_ui->comboBoxFrameDst->count() - 1)
        m_ui->comboBoxFrameDst->setCurrentIndex(index + 1);
}

void ToolWindowICPMatcher::updateWidgets()
{
    m_ui->actionLoad_Data_Set->setEnabled(!m_isLoaded);
    m_ui->actionMatch->setEnabled(m_isLoaded);
    m_ui->actionStep_Reset->setEnabled(m_isLoaded);
    m_ui->actionStep->setEnabled(m_isStepMode);
    m_ui->actionStep_GPU->setEnabled(m_isStepMode);
    m_ui->actionCompute_GPU->setEnabled(m_isLoaded);

    if (m_isInit)
    {
        m_ui->lineEditPairs->setText(QString("%1 of (%2, %3)").arg(m_pairs).arg(m_cloudSrc->points.size()).arg(m_cloudDst->points.size()));
    }
    m_ui->lineEditIteration->setText(QString::number(m_iteration));
    //m_ui->labelIteration->setText(QString::number(m_iteration));
    //m_ui->labelRotationError->setText(QString::number(qRadiansToDegrees(m_rotationError)));
    //m_ui->labelTranslationError->setText(QString::number(m_translationError));
}
