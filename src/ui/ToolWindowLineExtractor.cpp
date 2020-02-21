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
    m_cloudViewer2 = new CloudViewer(this);
    m_cloudViewer3 = new CloudViewer(this);

    m_ui->layoutPointCloud->addWidget(m_cloudViewer1);
    m_ui->layoutSecondary->addWidget(m_cloudViewer2);
    m_ui->layoutSecondary->addWidget(m_cloudViewer3);

    m_cloudViewer1->setCameraPosition(0, 0, -1.5f, 0, 0, 0, 0, -1, 0);
    m_cloudViewer2->setCameraPosition(0, 0, 1.5f, 0, 0, 0, 1, 0, 0);

    connect(m_ui->actionLoad_Point_Cloud, &QAction::triggered, this, &ToolWindowLineExtractor::onActionLoadPointCloud);
    connect(m_ui->actionGenerate_Line_Point_Cloud, &QAction::triggered, this, &ToolWindowLineExtractor::onActionGenerateLinePointCloud);
    connect(m_ui->actionParameterized_Points_Analysis, &QAction::triggered, this, &ToolWindowLineExtractor::onActionParameterizedPointsAnalysis);
    connect(m_ui->actionSave_Config, &QAction::triggered, this, &ToolWindowLineExtractor::onActionSaveConfig);
    connect(m_ui->actionLoad_Data_Set, &QAction::triggered, this, &ToolWindowLineExtractor::onActionLoadDataSet);
    connect(m_ui->pushButtonShowLineChain, &QPushButton::clicked, this, &ToolWindowLineExtractor::onActionShowLineChain);
    connect(m_ui->actionCompute_GPU, &QAction::triggered, this, &ToolWindowLineExtractor::onActionComputeGPU);

    m_ui->doubleSpinBoxSearchRadius->setValue(PARAMETERS.floatValue("search_radius", 0.05f, "LineExtractor"));
    m_ui->spinBoxMinNeighbours->setValue(PARAMETERS.intValue("min_neighbours", 3, "LineExtractor"));
    m_ui->doubleSpinBoxSearchErrorThreshold->setValue(PARAMETERS.floatValue("search_error_threshold", 0.025f, "LineExtractor"));
    m_ui->doubleSpinBoxAngleSearchRadius->setValue(qRadiansToDegrees(PARAMETERS.floatValue("angle_search_radius", qDegreesToRadians(20.0) * M_1_PI, "LineExtractor") * M_PI));
    m_ui->spinBoxAngleMinNeighbours->setValue(PARAMETERS.intValue("angle_min_neighbours", 10, "LineExtractor"));
    m_ui->doubleSpinBoxClusterTolerance->setValue(PARAMETERS.floatValue("mapping_tolerance", 0.01f, "LineExtractor"));
    m_ui->comboBoxAngleMappingMethod->setCurrentIndex(PARAMETERS.intValue("angle_mapping_method", 0, "LineExtractor"));
    m_ui->doubleSpinBoxMinLineLength->setValue(PARAMETERS.floatValue("min_line_length", 0.01f, "LineExtractor"));
    m_ui->doubleSpinBoxZDistanceThreshold->setValue(PARAMETERS.floatValue("region_growing_z_distance_threshold", 0.005f, "LineExtractor"));
    m_ui->doubleSpinBoxMSLRadiusSearch->setValue(PARAMETERS.floatValue("msl_radius_search", 0.01f, "LineExtractor"));
}

ToolWindowLineExtractor::~ToolWindowLineExtractor()
{
}

void ToolWindowLineExtractor::showLines()
{
    m_cloudViewer1->visualizer()->removeAllShapes();
    int lineNo = m_ui->comboBoxLineChains->currentIndex();
    if (lineNo < 0)
        return;
    qDebug() << "showLines:" << lineNo << m_chains.size();
    LineChain lc = m_chains[lineNo];

    /*for (int i = 0; i < m_chains.size(); i++)
    {
        LineChain& lc0 = m_chains[i];
        QString planeName = QString("plane_%1").arg(i);
        m_cloudViewer1->visualizer()->addPlane(*lc0.plane, lc0.point.x(), lc0.point.y(), lc0.point.z(), planeName.toStdString());
    }*/

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
        m_cloudViewer1->visualizer()->addLine(start, end, r, g, b, lineNo);
        m_cloudViewer1->visualizer()->addText3D(std::to_string(i), middle, 0.05, 255, 0, 0, textNo);
        m_cloudViewer1->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 4, lineNo);
    }

    for (int i = 0; i < m_mslCloud->size(); i++)
    {
        Line msl = m_mslCloud->points[i];
        pcl::PointXYZ start, end, middle;
        start.getVector3fMap() = msl.getEndPoint(-3);
        end.getVector3fMap() = msl.getEndPoint(3);
        middle.getVector3fMap() = msl.point;
        QString lineName = QString("msl_%1").arg(i);
        std::string textNo = "text_" + std::to_string(i);

        QColor color(0, 0, 255);
        int width = 1;
        if (i == lc.lineNo1)
        {
            color = QColor(255, 0, 0);
            width = 3;
        }
        else if (i == lc.lineNo2)
        {
            color = QColor(0, 0, 255);
            width = 3;
        }

        qDebug() << lineName;

        m_cloudViewer1->visualizer()->addLine(start, end, color.red(), color.green(), color.blue(), lineName.toStdString());
        m_cloudViewer1->visualizer()->addText3D(std::to_string(i), middle, 0.05, 1, 1, 1, textNo);
        m_cloudViewer1->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, width, lineName.toStdString());
    }

    qDebug() << "line chain" << lc.name();
    {
        pcl::PointXYZI start, end;
        start.getVector3fMap() = lc.point1;
        end.getVector3fMap() = lc.point2;
        m_cloudViewer1->visualizer()->addLine(start, end, 0, 255, 0, lc.name().toStdString());
        m_cloudViewer1->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, lc.name().toStdString());

        end.getVector3fMap() = lc.point;
        start.getVector3fMap() = lc.point + lc.xLocal * 0.2f;
        m_cloudViewer1->visualizer()->addArrow(start, end, 255, 0, 0, false, "xaxis");
        start.getVector3fMap() = lc.point + lc.yLocal * 0.2f;
        m_cloudViewer1->visualizer()->addArrow(start, end, 0, 255, 0, false, "yaxis");
        start.getVector3fMap() = lc.point + lc.zLocal * 0.2f;
        m_cloudViewer1->visualizer()->addArrow(start, end, 0, 0, 255, false, "zaxis");
    }
    
}

void ToolWindowLineExtractor::init()
{
    m_lineExtractor.reset(new LineExtractor);
    m_lineExtractor->setSearchRadius(m_ui->doubleSpinBoxSearchRadius->value());
    m_lineExtractor->setMinNeighbours(m_ui->spinBoxMinNeighbours->value());
    m_lineExtractor->setSearchErrorThreshold(m_ui->doubleSpinBoxSearchErrorThreshold->value());
    m_lineExtractor->setAngleSearchRadius(qDegreesToRadians(m_ui->doubleSpinBoxAngleSearchRadius->value()) * M_1_PI);
    m_lineExtractor->setAngleMinNeighbours(m_ui->spinBoxAngleMinNeighbours->value());
    m_lineExtractor->setMappingTolerance(m_ui->doubleSpinBoxClusterTolerance->value());
    m_lineExtractor->setAngleMappingMethod(m_ui->comboBoxAngleMappingMethod->currentIndex());
    m_lineExtractor->setMinLineLength(m_ui->doubleSpinBoxMinLineLength->value());
    m_lineExtractor->setRegionGrowingZDistanceThreshold(m_ui->doubleSpinBoxZDistanceThreshold->value());
    m_lineExtractor->setMslRadiusSearch(m_ui->doubleSpinBoxMSLRadiusSearch->value());

    m_boundaryExtractor.reset(new BoundaryExtractor);
    m_boundaryExtractor->setDownsamplingMethod(PARAMETERS.intValue("downsampling_method", 0, "BoundaryExtractor"));
    m_boundaryExtractor->setEnableRemovalFilter(PARAMETERS.boolValue("enable_removal_filter", false, "BoundaryExtractor"));
    m_boundaryExtractor->setDownsampleLeafSize(PARAMETERS.floatValue("downsample_leaf_size", 0.0075f, "BoundaryExtractor"));
    m_boundaryExtractor->setOutlierRemovalMeanK(PARAMETERS.floatValue("outlier_removal_mean_k", 20.f, "BoundaryExtractor"));
    m_boundaryExtractor->setStddevMulThresh(PARAMETERS.floatValue("std_dev_mul_thresh", 1.f, "BoundaryExtractor"));
    m_boundaryExtractor->setGaussianSigma(PARAMETERS.floatValue("gaussian_sigma", 4.f, "BoundaryExtractor"));
    m_boundaryExtractor->setGaussianRSigma(PARAMETERS.floatValue("gaussian_r_sigma", 4.f, "BoundaryExtractor"));
    m_boundaryExtractor->setGaussianRadiusSearch(PARAMETERS.floatValue("gaussian_radius_search", 0.05f, "BoundaryExtractor"));
    m_boundaryExtractor->setNormalsRadiusSearch(PARAMETERS.floatValue("normals_radius_search", 0.05f, "BoundaryExtractor"));
    m_boundaryExtractor->setBoundaryRadiusSearch(PARAMETERS.floatValue("boundary_radius_search", 0.1f, "BoundaryExtractor"));
    m_boundaryExtractor->setBoundaryAngleThreshold(PARAMETERS.floatValue("boundary_angle_threshold", M_PI_2, "BoundaryExtractor"));
    m_boundaryExtractor->setBorderLeft(PARAMETERS.floatValue("border_left", 26, "BoundaryExtractor"));
    m_boundaryExtractor->setBorderRight(PARAMETERS.floatValue("border_right", 22, "BoundaryExtractor"));
    m_boundaryExtractor->setBorderTop(PARAMETERS.floatValue("border_top", 16, "BoundaryExtractor"));
    m_boundaryExtractor->setBorderBottom(PARAMETERS.floatValue("border_bottom", 16, "BoundaryExtractor"));
    m_boundaryExtractor->setProjectedRadiusSearch(qDegreesToRadians(PARAMETERS.floatValue("projected_radius_search", 5, "BoundaryExtractor")));
    m_boundaryExtractor->setVeilDistanceThreshold(PARAMETERS.floatValue("veil_distance_threshold", 0.1f, "BoundaryExtractor"));
    m_boundaryExtractor->setPlaneDistanceThreshold(PARAMETERS.floatValue("plane_distance_threshold", 0.01f, "BoundaryExtractor"));
}

void ToolWindowLineExtractor::compute()
{
    init();
    if (m_fromDataSet)
    {
        int frameIndex = m_ui->comboBoxFrameIndex->currentIndex();
        Frame frame = m_device->getFrame(frameIndex);
        pcl::IndicesPtr indices(new std::vector<int>);
        m_dataCloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZI>::Ptr boundaryPoints;
        m_cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);

        m_ui->widgetRGBFrame->setImage(cvMat2QImage(frame.colorMat(), true));

        m_colorCloud = frame.getCloud(*indices);
        pcl::copyPointCloud<pcl::PointXYZRGB, pcl::PointXYZ>(*m_colorCloud, *m_dataCloud);

        m_boundaryExtractor->setInputCloud(m_dataCloud);
        m_boundaryExtractor->setMatWidth(frame.getDepthWidth());
        m_boundaryExtractor->setMatHeight(frame.getDepthHeight());
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
            m_boundaryExtractor->compute();
        }
        m_cloud = m_boundaryExtractor->boundaryPoints();
        m_filteredCloud = m_boundaryExtractor->filteredCloud();
        m_planes = m_boundaryExtractor->planes();
    }

    m_lines = m_lineExtractor->compute(m_cloud);
    if (m_fromDataSet)
    {
        m_lineExtractor->extractLinesFromPlanes(m_planes);
        m_lineExtractor->segmentLines();
    }
    m_lineExtractor->generateLineChains();
    //m_lineExtractor->generateDescriptors();
    m_lineExtractor->generateDescriptors2();

    m_chains = m_lineExtractor->chains();
    pcl::PointCloud<pcl::PointXYZI>::Ptr angleCloud = m_lineExtractor->angleCloud();
    QList<float> densityList = m_lineExtractor->densityList();
    QList<int> angleCloudIndices = m_lineExtractor->angleCloudIndices();
    QMap<int, std::vector<int>> subCloudIndices = m_lineExtractor->subCloudIndices();
    pcl::PointCloud<pcl::PointXYZI>::Ptr mappingCloud = m_lineExtractor->mappingCloud();
    QList<float> errors = m_lineExtractor->errors();
    pcl::PointCloud<pcl::PointXYZI>::Ptr linedCloud = m_lineExtractor->linedCloud();
    QList<int> linePointsCount = m_lineExtractor->linePointsCount();
    m_mslCloud = m_lineExtractor->mslCloud();

    m_cloudViewer1->visualizer()->removeAllPointClouds();
    m_cloudViewer1->visualizer()->removeAllShapes();
    m_cloudViewer2->visualizer()->removeAllPointClouds();
    m_cloudViewer2->visualizer()->removeAllShapes();
    m_cloudViewer3->visualizer()->removeAllPointClouds();
    m_cloudViewer3->visualizer()->removeAllShapes();

    m_ui->comboBoxLineChains->clear();
    for (int i = 0; i < m_chains.size(); i++)
    {
        m_ui->comboBoxLineChains->addItem(QString("%1[%2-%3]").arg(i).arg(m_chains[i].lineNo1).arg(m_chains[i].lineNo2));
    }
    m_ui->comboBoxLineChains->setCurrentIndex(0);

    pcl::PointCloud<pcl::PointXYZI>::Ptr densityCloud(new pcl::PointCloud<pcl::PointXYZI>);
    for (int i = 0; i < densityList.size(); i++)
    {
        int index = angleCloudIndices[i];
        //qDebug() << indexList[i] << m_density[indexList[i]];

        pcl::PointXYZI ptAngle = angleCloud->points[index];
        ptAngle.intensity = densityList[index];
        densityCloud->push_back(ptAngle);
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr boundaryCloud2(new pcl::PointCloud<pcl::PointXYZI>);
    for (QMap<int, std::vector<int>>::iterator i = subCloudIndices.begin(); i != subCloudIndices.end(); i++)
    {
        int index = i.key();
        pcl::PointXYZI pt = m_cloud->points[index];
        pt.intensity = index;
        int color = qrand();

        for (std::vector<int>::iterator itNeighbour = i.value().begin(); itNeighbour != i.value().end(); itNeighbour++)
        {
            int neighbourIndex = *itNeighbour;
            pcl::PointXYZI ptNeighbour = m_cloud->points[neighbourIndex];
            ptNeighbour.intensity = color;
            boundaryCloud2->push_back(ptNeighbour);
        }
    }

    if (m_ui->radioButtonShowBoundaryCloud->isChecked())
    {
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> iColor(m_cloud, "intensity");
        m_cloudViewer1->visualizer()->addPointCloud(m_cloud, iColor, "original cloud");
        m_cloudViewer1->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "original cloud");
    }
    else if (m_ui->radioButtonShowGroupedCloud->isChecked())
    {
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> iColor(boundaryCloud2, "intensity");
        m_cloudViewer1->visualizer()->addPointCloud(boundaryCloud2, iColor, "grouped cloud");
        m_cloudViewer1->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "grouped cloud");
    }
    else if (m_ui->radioButtonShowLinedCloud->isChecked())
    {
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> iColor(linedCloud, "intensity");
        m_cloudViewer1->visualizer()->addPointCloud(linedCloud, iColor, "lined cloud");
        m_cloudViewer1->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "lined cloud");
    }

    {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> iColor(m_filteredCloud, 127, 127, 127);
        m_cloudViewer1->visualizer()->addPointCloud(m_filteredCloud, iColor, "filtered cloud");
    }

    if (m_ui->radioButtonShowAngleCloud->isChecked())
    {
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> iColor(angleCloud, "intensity");
        m_cloudViewer2->visualizer()->addPointCloud(angleCloud, iColor, "angle cloud");
        m_cloudViewer2->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "angle cloud");
    }
    else if (m_ui->radioButtonShowDensityCloud->isChecked())
    {
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> iColor(densityCloud, "intensity");
        m_cloudViewer2->visualizer()->addPointCloud(densityCloud, iColor, "density cloud");
        m_cloudViewer2->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "density cloud");
    }

    {
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> iColor(mappingCloud, "intensity");
        m_cloudViewer3->visualizer()->addPointCloud(mappingCloud, iColor, "mapping cloud");
        m_cloudViewer3->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "mapping cloud");
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

void ToolWindowLineExtractor::onActionSaveConfig()
{
    PARAMETERS.setValue("search_radius", m_ui->doubleSpinBoxSearchRadius->value(), "LineExtractor");
    PARAMETERS.setValue("min_neighbours", m_ui->spinBoxMinNeighbours->value(), "LineExtractor");
    PARAMETERS.setValue("search_error_threshold", m_ui->doubleSpinBoxSearchErrorThreshold->value(), "LineExtractor");
    PARAMETERS.setValue("angle_search_radius", qDegreesToRadians(m_ui->doubleSpinBoxAngleSearchRadius->value()) * M_1_PI, "LineExtractor");
    PARAMETERS.setValue("angle_min_neighbours", m_ui->spinBoxAngleMinNeighbours->value(), "LineExtractor");
    PARAMETERS.setValue("mapping_tolerance", m_ui->doubleSpinBoxClusterTolerance->value(), "LineExtractor");
    PARAMETERS.setValue("angle_mapping_method", m_ui->comboBoxAngleMappingMethod->currentIndex(), "LineExtractor");
    PARAMETERS.setValue("min_line_length", m_ui->doubleSpinBoxMinLineLength->value(), "LineExtractor");
    PARAMETERS.setValue("region_growing_z_distance_threshold", m_ui->doubleSpinBoxZDistanceThreshold->value(), "LineExtractor");
    PARAMETERS.setValue("msl_radius_search", m_ui->doubleSpinBoxMSLRadiusSearch->value(), "LineExtractor");

    PARAMETERS.save();
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
        if (m_ui->checkBoxEnableRandom->isChecked())
        {
            offsetX = qrand() * 1.f / RAND_MAX * m_ui->doubleSpinBoxRandomOffset->value();
            offsetY = qrand() * 1.f / RAND_MAX * m_ui->doubleSpinBoxRandomOffset->value();
            offsetZ = qrand() * 1.f / RAND_MAX * m_ui->doubleSpinBoxRandomOffset->value();
        }
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

void ToolWindowLineExtractor::onActionGenerateLinePointCloud()
{
    m_cloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    Eigen::Vector3f start1(0.0f, 0.f, 0.1f);
    Eigen::Vector3f start2(0.0f, 0.f, 0.1f);
    Eigen::Vector3f start3(0.3f, 0.f, -0.1f);
    Eigen::Vector3f dir1(0, 1.f, 0.f);
    Eigen::Vector3f dir2(1, 0.f, 0.f);
    Eigen::Vector3f dir3(1, 1.f, 1.f);
    dir1.normalize();
    dir2.normalize();
    dir3.normalize();

    m_cloudViewer1->visualizer()->removeAllPointClouds();
    m_cloudViewer1->visualizer()->removeAllShapes();

    qsrand(QDateTime::currentMSecsSinceEpoch());

    for (int i = 0; i < 3000; i++)
    {
        if (i < 1000)
        {
            Eigen::Vector3f ePt = start1 + dir1 * i * 0.001f;
            pcl::PointXYZI point;
            float offsetX = qrand() * 1.f / RAND_MAX * m_ui->doubleSpinBoxRandomOffset->value();
            float offsetY = qrand() * 1.f / RAND_MAX * m_ui->doubleSpinBoxRandomOffset->value();
            float offsetZ = qrand() * 1.f / RAND_MAX * m_ui->doubleSpinBoxRandomOffset->value();
            point.x = ePt.x() + offsetX;
            point.y = ePt.y() + offsetY;
            point.z = ePt.z() + offsetZ;
            point.intensity = i;
            m_cloud->push_back(point);
        }
        else if (i >= 1000 && i < 2000)
        {
            continue;
            Eigen::Vector3f ePt = start2 + dir2 *(i - 1000) * 0.001f;
            pcl::PointXYZI point;
            float offsetX = qrand() * 1.f / RAND_MAX * m_ui->doubleSpinBoxRandomOffset->value();
            float offsetY = qrand() * 1.f / RAND_MAX * m_ui->doubleSpinBoxRandomOffset->value();
            float offsetZ = qrand() * 1.f / RAND_MAX * m_ui->doubleSpinBoxRandomOffset->value();
            point.x = ePt.x() + offsetX;
            point.y = ePt.y() + offsetY;
            point.z = ePt.z() + offsetZ;
            point.intensity = i;
            m_cloud->push_back(point);
        }
        else if (i >= 2000)
        {
            Eigen::Vector3f ePt = start3 + dir3 *(i - 2000) * 0.001f;
            pcl::PointXYZI point;
            float offsetX = qrand() * 1.f / RAND_MAX * m_ui->doubleSpinBoxRandomOffset->value();
            float offsetY = qrand() * 1.f / RAND_MAX * m_ui->doubleSpinBoxRandomOffset->value();
            float offsetZ = qrand() * 1.f / RAND_MAX * m_ui->doubleSpinBoxRandomOffset->value();
            point.x = ePt.x() + offsetX;
            point.y = ePt.y() + offsetY;
            point.z = ePt.z() + offsetZ;
            point.intensity = i;
            m_cloud->push_back(point);
        }
    }
    m_cloud->is_dense = true;
    qDebug() << "cloud size:" << m_cloud->size();

    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> iColor(m_cloud, "intensity");
    m_cloudViewer1->visualizer()->addPointCloud(m_cloud, iColor, "original cloud");
    m_cloudViewer1->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "original cloud");
}
