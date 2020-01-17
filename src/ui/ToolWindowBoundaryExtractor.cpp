#include "ToolWindowBoundaryExtractor.h"
#include "ui_ToolWindowBoundaryExtractor.h"

#include <QDebug>
#include <QAction>
#include <QFileDialog>
#include <QDir>

#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>

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

    m_ui->verticalLayout1->addWidget(m_cloudViewer);

    connect(m_ui->actionLoad_Data_Set, &QAction::triggered, this, &ToolWindowBoundaryExtractor::onActionLoadDataSet);
    connect(m_ui->actionCompute, &QAction::triggered, this, &ToolWindowBoundaryExtractor::onActionCompute);
    connect(m_ui->actionSave, &QAction::triggered, this, &ToolWindowBoundaryExtractor::onActionSave);
}

ToolWindowBoundaryExtractor::~ToolWindowBoundaryExtractor()
{
}

void ToolWindowBoundaryExtractor::onActionCompute()
{
    m_boundaryExtractor.reset(new BoundaryExtractor);
    m_boundaryExtractor->setDownsampleLeafSize(m_ui->doubleSpinBoxDownsampleLeafSize->value());
    m_boundaryExtractor->setOutlierRemovalMeanK(m_ui->doubleSpinBoxOutlierRemovalMeanK->value());
    m_boundaryExtractor->setStddevMulThresh(m_ui->doubleSpinBoxStddevMulThresh->value());
    m_boundaryExtractor->setGaussianSigma(m_ui->doubleSpinBoxGaussianSigma->value());
    m_boundaryExtractor->setGaussianRSigma(m_ui->doubleSpinBoxGaussianRSigma->value());
    m_boundaryExtractor->setGaussianRadiusSearch(m_ui->doubleSpinBoxGaussianRadiusSearch->value());
    m_boundaryExtractor->setNormalsRadiusSearch(m_ui->doubleSpinBoxNormalsRadiusSearch->value());
    m_boundaryExtractor->setBoundaryRadiusSearch(m_ui->doubleSpinBoxRadiusSearch->value());
    m_boundaryExtractor->setBoundaryAngleThreshold(M_PI / m_ui->spinBoxAngleThresholdDivision->value());

    m_cloudViewer->visualizer()->removeAllPointClouds();
    m_cloudViewer->visualizer()->removeAllShapes();

    int frameIndex = m_ui->comboBoxFrameIndex->currentIndex();
    Frame frame = m_device->getFrame(frameIndex);
    pcl::IndicesPtr indices(new pcl::Indices);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorCloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    colorCloud = frame.getCloud(*indices);
    pcl::copyPointCloud<pcl::PointXYZRGB, pcl::PointXYZ>(*colorCloud, *cloud);

    m_boundaryExtractor->setInputCloud(cloud);
    //m_boundaryExtractor->setIndices(indices);
    m_boundary = m_boundaryExtractor->compute();
    pcl::PointCloud<pcl::PointXYZ>::Ptr filteredCloud = m_boundaryExtractor->filteredCloud();
    pcl::PointCloud<pcl::Normal>::Ptr normalsCloud = m_boundaryExtractor->normals();

    if (m_ui->radioButtonShowColor->isChecked())
    {
        m_cloudViewer->addCloud("scene cloud", colorCloud);
    }
    else if (m_ui->radioButtonShowNoColor->isChecked())
    {
        m_cloudViewer->addCloud("scene cloud", filteredCloud);
    }

    if (m_ui->checkBoxShowNormals->isChecked())
    {
        m_cloudViewer->visualizer()->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(filteredCloud, normalsCloud, 10, 0.02f);
    }

    {
        pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> iColor(m_boundary, "intensity");
        m_cloudViewer->visualizer()->addPointCloud(m_boundary, iColor, "boundary cloud");
        m_cloudViewer->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "boundary cloud");
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
    pcl::copyPointCloud(*m_boundary, cloud);
    cloud.width = cloud.size();
    cloud.height = 1;
    cloud.is_dense = true;
    pcl::io::savePCDFile<pcl::PointXYZ>(fileName.toStdString(), cloud);
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
