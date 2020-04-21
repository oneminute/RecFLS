#include "ToolWindowLineExtractor.h"
#include "ui_ToolWindowLineExtractor.h"

#include <QDebug>
#include <QDateTime>
#include <QtMath>
#include <QFileDialog>
#include <QDir>
#include <QPushButton>

#include "common/Parameters.h"
#include "util/Utils.h"
#include "extractor/LineExtractor.h"

#include <pcl/io/ply_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/pca.h>

ToolWindowLineExtractor::ToolWindowLineExtractor(QWidget* parent)
    : QMainWindow(parent)
    , m_ui(new Ui::ToolWindowLineExtractor)
    , m_fromDataSet(false)
    , m_useCuda(false)
    , m_init(false)
{
    m_ui->setupUi(this);

    m_cloudViewer1 = new CloudViewer(this);
    //m_cloudViewer2 = new CloudViewer(this);
    //m_cloudViewer3 = new CloudViewer(this);

    m_ui->layoutPointCloud->addWidget(m_cloudViewer1);
    //m_ui->layoutSecondary->addWidget(m_cloudViewer2);
    //m_ui->layoutSecondary->addWidget(m_cloudViewer3);

    m_cloudViewer1->setCameraPosition(0, 0, -1.5f, 0, 0, 0, 0, -1, 0);
    //m_cloudViewer2->setCameraPosition(0, 0, 1.5f, 0, 0, 0, 1, 0, 0);

    connect(m_ui->actionLoad_Data_Set, &QAction::triggered, this, &ToolWindowLineExtractor::onActionLoadDataSet);
    connect(m_ui->actionCompute_GPU, &QAction::triggered, this, &ToolWindowLineExtractor::onActionComputeGPU);
}

ToolWindowLineExtractor::~ToolWindowLineExtractor()
{
}

void ToolWindowLineExtractor::showLines()
{
    m_cloudViewer1->visualizer()->removeAllShapes();

    for (int i = 0; i < m_lines.size()/* && errors[i]*/; i++)
    {
        double r = rand() * 1.0 / RAND_MAX;
        double g = rand() * 1.0 / RAND_MAX;
        double b = rand() * 1.0 / RAND_MAX;
        LineSegment line = m_lines[i];
        std::string lineNo = "line_" + std::to_string(i);
        std::string textNo = "ls_text_" + std::to_string(i);
        //qDebug() << QString::fromStdString(lineNo) << line.length() << errors[i] << linePointsCount[i];
        pcl::PointXYZI start, end, middle;
        start.getVector3fMap() = line.start();
        end.getVector3fMap() = line.end();
        Eigen::Vector3f dir = line.direction();
        middle.getVector3fMap() = line.middle();
        //m_cloudViewer->visualizer()->addArrow(end, start, r, g, b, 0, lineNo);
        m_cloudViewer1->visualizer()->addLine(start, end, 255, 255, 0, lineNo);
        m_cloudViewer1->visualizer()->addText3D(std::to_string(i), middle, 0.05, 255, 0, 0, textNo);
        m_cloudViewer1->visualizer()->addSphere(middle, 0.01f, 0, 255, 0, std::to_string(i) + "_sphere");
        m_cloudViewer1->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 6, lineNo);
    }

    for (int i = 0; i < m_lineCloud->size(); i++)
    {
        Line msl = m_lineCloud->points[i];
        pcl::PointXYZ start, end, middle;
        start.getVector3fMap() = msl.getEndPoint(-3);
        end.getVector3fMap() = msl.getEndPoint(3);
        middle.getVector3fMap() = msl.point;
        QString lineName = QString("msl_%1").arg(i);
        std::string textNo = "text_" + std::to_string(i);

        QColor color(0, 255, 0);
        int width = 1;
        
        qDebug() << lineName;

        m_cloudViewer1->visualizer()->addLine(start, end, color.red(), color.green(), color.blue(), lineName.toStdString());
        m_cloudViewer1->visualizer()->addText3D(std::to_string(i), middle, 0.05, 1, 1, 1, textNo);
        m_cloudViewer1->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, width, lineName.toStdString());
    }
}

void ToolWindowLineExtractor::init()
{
    m_lineExtractor.reset(new LineExtractor);

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
}

void ToolWindowLineExtractor::compute()
{
    init();
    cv::Mat board;
    if (m_fromDataSet)
    {
        int frameIndex = m_ui->comboBoxFrameIndex->currentIndex();
        Frame frame = m_device->getFrame(frameIndex);
        pcl::IndicesPtr indices(new std::vector<int>);
        m_dataCloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr boundaryPoints;
        m_cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);

        m_ui->widgetRGBFrame->setImage(cvMat2QImage(frame.colorMat(), true));
        board = frame.colorMat();

        m_colorCloud = frame.getCloud(*indices);
        pcl::copyPointCloud<pcl::PointXYZRGB, pcl::PointXYZ>(*m_colorCloud, *m_dataCloud);

        m_boundaryExtractor->setInputCloud(m_dataCloud);
        m_boundaryExtractor->setWidth(frame.getDepthWidth());
        m_boundaryExtractor->setHeight(frame.getDepthHeight());
        m_boundaryExtractor->setCx(frame.getDevice()->cx());
        m_boundaryExtractor->setCy(frame.getDevice()->cy());
        m_boundaryExtractor->setFx(frame.getDevice()->fx());
        m_boundaryExtractor->setFy(frame.getDevice()->fy());
        m_boundaryExtractor->setNormals(nullptr);

        if (m_useCuda)
        {
            if (!m_init)
            {
                m_frameGpu.parameters.colorWidth = frame.getColorWidth();
                m_frameGpu.parameters.colorHeight = frame.getColorHeight();
                m_frameGpu.parameters.depthWidth = frame.getDepthWidth();
                m_frameGpu.parameters.depthHeight = frame.getDepthHeight();
                m_frameGpu.allocate();
                m_init = true;
            }

            cv::cuda::GpuMat colorMatGpu(frame.getColorHeight(), frame.getColorWidth(), CV_8UC3, m_frameGpu.colorImage);
            cv::cuda::GpuMat depthMatGpu(frame.getDepthHeight(), frame.getDepthWidth(), CV_16U, m_frameGpu.depthImage);
            colorMatGpu.upload(frame.colorMat());
            depthMatGpu.upload(frame.depthMat());

            m_boundaryExtractor->computeCUDA(m_frameGpu);
        }
        else
        {
            //m_boundaryExtractor->compute();
        }
        m_originalCloud = m_boundaryExtractor->cloud();
        m_cloud = m_boundaryExtractor->boundaryPoints();
        m_filteredCloud = m_boundaryExtractor->filteredCloud();
        m_planes = m_boundaryExtractor->planes();
        pcl::PointCloud<pcl::PointXYZI>::Ptr cornerPoints = m_boundaryExtractor->cornerPoints();

        m_lines = m_lineExtractor->compute(m_cloud, cornerPoints);
    }

    m_lineCloud = m_lineExtractor->lineCloud();
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud = m_lineExtractor->cloud();

    m_cloudViewer1->visualizer()->removeAllPointClouds();
    m_cloudViewer1->visualizer()->removeAllShapes();
    //m_cloudViewer2->visualizer()->removeAllPointClouds();
    //m_cloudViewer2->visualizer()->removeAllShapes();
    //m_cloudViewer3->visualizer()->removeAllPointClouds();
    //m_cloudViewer3->visualizer()->removeAllShapes(); 

    pcl::PointCloud<pcl::PointXYZI>::Ptr densityCloud(new pcl::PointCloud<pcl::PointXYZI>);
    
    {
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> iColor(cloud, "intensity");
        m_cloudViewer1->visualizer()->addPointCloud(cloud, iColor, "points cloud");
        m_cloudViewer1->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "points cloud");
    }

    {
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgbHandler(m_colorCloud);
        m_cloudViewer1->visualizer()->addPointCloud(m_colorCloud, rgbHandler, "scene cloud");
        //pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> iColor(m_originalCloud, 127, 127, 127);
        //m_cloudViewer1->visualizer()->addPointCloud(m_originalCloud, iColor, "original cloud");
    }

    showLines();
}

void ToolWindowLineExtractor::onActionParameterizedPointsAnalysis()
{
    m_useCuda = false;
    compute();
}

void ToolWindowLineExtractor::onActionComputeGPU()
{
    m_useCuda = true;
    compute();
}

void ToolWindowLineExtractor::onActionShowLineChain(bool checked)
{
    showLines();
}

void ToolWindowLineExtractor::onActionLoadDataSet()
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
    m_ui->comboBoxFrameIndex->setCurrentIndex(0);
    m_fromDataSet = true;
}

void ToolWindowLineExtractor::onActionLoadPointCloud()
{
    QString fileName = QFileDialog::getOpenFileName(this, tr("Point Cloud"), QDir::current().absolutePath(), tr("Polygon Files (*.obj *.ply *.pcd)"));
    if (fileName.isEmpty())
    {
        return;
    }
    QFileInfo info(fileName);

    m_cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
    if (info.completeSuffix().contains("obj"))
    {
        pcl::io::loadOBJFile<pcl::PointXYZ>(fileName.toStdString(), *cloud);
    }
    else if (info.completeSuffix().contains("ply"))
    {
        pcl::io::loadPLYFile<pcl::PointXYZ>(fileName.toStdString(), *cloud);
    }
    else if (info.completeSuffix().contains("pcd"))
    {
        pcl::io::loadPCDFile<pcl::PointXYZ>(fileName.toStdString(), *cloud);
    }
    qDebug() << fileName << cloud->size();

    qsrand(QDateTime::currentMSecsSinceEpoch());
    
    for (int i = 0; i < cloud->size(); i++)
    {
        pcl::PointXYZ inPt = cloud->points[i];
        pcl::PointXYZI outPt;
        float offsetX = 0;
        float offsetY = 0;
        float offsetZ = 0;
        
        outPt.x = inPt.x + offsetX;
        outPt.y = inPt.y + offsetY;
        outPt.z = inPt.z + offsetZ;
        outPt.intensity = i;
        m_cloud->push_back(outPt);
    }

    m_cloudViewer1->visualizer()->removeAllPointClouds();
    m_cloudViewer1->visualizer()->removeAllShapes();
    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> iColor(m_cloud, "intensity");
    m_cloudViewer1->visualizer()->addPointCloud(m_cloud, iColor, "original cloud");
    m_cloudViewer1->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original cloud");

    m_fromDataSet = false;
}

