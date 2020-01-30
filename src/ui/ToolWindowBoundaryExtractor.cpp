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
{
    m_ui->setupUi(this);

    m_cloudViewer = new CloudViewer;
    m_cloudViewer->setCameraPosition(0, 0, -1.5f, 0, 0, 0, 0, -1, 0);
    m_projectedCloudViewer = new CloudViewer;
    m_projectedCloudViewer->setCameraPosition(0, 0, -1.5f, 0, 0, 0, 0, 1, 0);
    m_depthViewer = new ImageViewer;

    m_ui->verticalLayout1->addWidget(m_cloudViewer);
    m_ui->verticalLayout2->addWidget(m_projectedCloudViewer);
    m_ui->verticalLayout2->addWidget(m_depthViewer);

    //m_ui->comboBoxDownsamplingMethod->setCurrentIndex(PARAMETERS.intValue();

    connect(m_ui->actionLoad_Data_Set, &QAction::triggered, this, &ToolWindowBoundaryExtractor::onActionLoadDataSet);
    connect(m_ui->actionCompute, &QAction::triggered, this, &ToolWindowBoundaryExtractor::onActionCompute);
    connect(m_ui->actionSave, &QAction::triggered, this, &ToolWindowBoundaryExtractor::onActionSave);
    connect(m_ui->actionSave_Config, &QAction::triggered, this, &ToolWindowBoundaryExtractor::onActionSaveConfig);
}

ToolWindowBoundaryExtractor::~ToolWindowBoundaryExtractor()
{
}

void ToolWindowBoundaryExtractor::onActionCompute()
{
    m_boundaryExtractor.reset(new BoundaryExtractor);
    m_boundaryExtractor->setDownsamplingMethod(m_ui->comboBoxDownsamplingMethod->currentIndex());
    m_boundaryExtractor->setEnableRemovalFilter(m_ui->checkBoxEnableRemovalFilter->isChecked());
    m_boundaryExtractor->setDownsampleLeafSize(m_ui->doubleSpinBoxDownsampleLeafSize->value());
    m_boundaryExtractor->setOutlierRemovalMeanK(m_ui->doubleSpinBoxOutlierRemovalMeanK->value());
    m_boundaryExtractor->setStddevMulThresh(m_ui->doubleSpinBoxStddevMulThresh->value());
    m_boundaryExtractor->setGaussianSigma(m_ui->doubleSpinBoxGaussianSigma->value());
    m_boundaryExtractor->setGaussianRSigma(m_ui->doubleSpinBoxGaussianRSigma->value());
    m_boundaryExtractor->setGaussianRadiusSearch(m_ui->doubleSpinBoxGaussianRadiusSearch->value());
    m_boundaryExtractor->setNormalsRadiusSearch(m_ui->doubleSpinBoxNormalsRadiusSearch->value());
    m_boundaryExtractor->setBoundaryRadiusSearch(m_ui->doubleSpinBoxRadiusSearch->value());
    m_boundaryExtractor->setBoundaryAngleThreshold(M_PI / m_ui->spinBoxAngleThresholdDivision->value());
    m_boundaryExtractor->setBorderLeft(m_ui->spinBoxBorderLeft->value());
    m_boundaryExtractor->setBorderRight(m_ui->spinBoxBorderRight->value());
    m_boundaryExtractor->setBorderTop(m_ui->spinBoxBorderTop->value());
    m_boundaryExtractor->setBorderBottom(m_ui->spinBoxBorderBottom->value());
    m_boundaryExtractor->setProjectedRadiusSearch(qDegreesToRadians(m_ui->doubleSpinBoxProjectedRadiusSearch->value()));
    m_boundaryExtractor->setVeilDistanceThreshold(m_ui->doubleSpinBoxVeilDistanceThreshold->value());

    m_cloudViewer->visualizer()->removeAllPointClouds();
    m_cloudViewer->visualizer()->removeAllShapes();
    m_projectedCloudViewer->visualizer()->removeAllPointClouds();
    m_projectedCloudViewer->visualizer()->removeAllShapes();

    int frameIndex = m_ui->comboBoxFrameIndex->currentIndex();
    Frame frame = m_device->getFrame(frameIndex);
    pcl::IndicesPtr indices(new pcl::Indices);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorCloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    colorCloud = frame.getCloud(*indices);
    pcl::copyPointCloud<pcl::PointXYZRGB, pcl::PointXYZ>(*colorCloud, *cloud);

    m_boundaryExtractor->setInputCloud(cloud);
    m_boundaryExtractor->setMatWidth(frame.getDepthWidth());
    m_boundaryExtractor->setMatHeight(frame.getDepthHeight());
    m_boundaryExtractor->setCx(frame.getDevice()->cx());
    m_boundaryExtractor->setCy(frame.getDevice()->cy());
    m_boundaryExtractor->setFx(frame.getDevice()->fx());
    m_boundaryExtractor->setFy(frame.getDevice()->fy());
    //m_boundaryExtractor->setIndices(indices);
    m_allBoundary = m_boundaryExtractor->compute();
    pcl::PointCloud<pcl::PointXYZ>::Ptr filteredCloud = m_boundaryExtractor->filteredCloud();
    pcl::PointCloud<pcl::Normal>::Ptr normalsCloud = m_boundaryExtractor->normals();

    pcl::PointCloud<pcl::PointXYZI>::Ptr m_projectedCloud = m_boundaryExtractor->projectedCloud();
    m_boundaryPoints = m_boundaryExtractor->boundaryPoints();
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_veilPoints = m_boundaryExtractor->veilPoints();
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_borderPoints = m_boundaryExtractor->borderPoints();

    m_depthViewer->setImage(cvMat2QImage(frame.depthMat(), false));
    if (m_ui->radioButtonShowColor->isChecked())
    {
        m_cloudViewer->addCloud("scene cloud", colorCloud);
    }
    else if (m_ui->radioButtonShowNoColor->isChecked())
    {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> yColor(filteredCloud, 255, 255, 127);
        m_cloudViewer->visualizer()->addPointCloud(filteredCloud, yColor, "scene cloud");
    }

    if (m_ui->checkBoxShowNormals->isChecked())
    {
        m_cloudViewer->visualizer()->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(filteredCloud, normalsCloud, 10, 0.02f);
    }

    {
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> iColor(m_projectedCloud, "intensity");
        m_projectedCloudViewer->visualizer()->addPointCloud(m_projectedCloud, iColor, "projected cloud");
        m_projectedCloudViewer->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "projected cloud");
    }

    {
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> rColor(m_boundaryPoints, 255, 0, 0);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> gColor(m_veilPoints, 0, 255, 0);
        pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> bColor(m_borderPoints, 0, 0, 255);

        m_cloudViewer->visualizer()->addPointCloud(m_boundaryPoints, rColor, "boundary points");
        m_cloudViewer->visualizer()->addPointCloud(m_veilPoints, gColor, "veil points");
        m_cloudViewer->visualizer()->addPointCloud(m_borderPoints, bColor, "border points");
        m_cloudViewer->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "boundary points");
        m_cloudViewer->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "veil points");
        m_cloudViewer->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "border points");

        pcl::PointXYZI zero;
        zero.getVector3fMap() = Eigen::Vector3f::Zero();
        for (int i = 0; i < m_veilPoints->size(); i++)
        {
            if (i % 10 == 0)
            {
                m_cloudViewer->visualizer()->addLine<pcl::PointXYZI, pcl::PointXYZI>(zero, m_veilPoints->points[i], "line_" + std::to_string(i));
            }
        }
    }

    StopWatch::instance().debugPrint();
}

void ToolWindowBoundaryExtractor::onActionSave()
{
    QString fileName = QString("%1/%2_%3_boundary.pcd").arg(QDir::currentPath())
        .arg(Parameters::Global().stringValue("sample_path", "samples/office3.sens", "Device_SensorReader")).arg(m_ui->comboBoxFrameIndex->currentIndex());
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
    PARAMETERS.setValue("downsampling_method", m_ui->comboBoxDownsamplingMethod->currentIndex(), "BoundaryExtractor");
    PARAMETERS.setValue("enable_removal_filter", m_ui->checkBoxEnableRemovalFilter->isChecked(), "BoundaryExtractor");
    PARAMETERS.setValue("downsample_leaf_size", m_ui->doubleSpinBoxDownsampleLeafSize->value(), "BoundaryExtractor");
    PARAMETERS.setValue("outlier_removal_mean_k", m_ui->doubleSpinBoxOutlierRemovalMeanK->value(), "BoundaryExtractor");
    PARAMETERS.setValue("std_dev_mul_thresh", m_ui->doubleSpinBoxStddevMulThresh->value(), "BoundaryExtractor");
    PARAMETERS.setValue("gaussian_sigma", m_ui->doubleSpinBoxGaussianSigma->value(), "BoundaryExtractor");
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

    PARAMETERS.save();
}

void ToolWindowBoundaryExtractor::onActionLoadDataSet()
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
}
