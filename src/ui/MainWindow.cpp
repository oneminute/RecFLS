#include "MainWindow.h"
#include "ui_MainWindow.h"

#include <pcl/common/pca.h>
#include <pcl/filters/filter.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "PreferencesWindow.h"
#include "ToolWindowBoundaryExtractor.h"
#include "ToolWindowFusedLineExtractor.h"
#include "ToolWindowLineExtractor.h"
#include "ToolWindowLineMatcher.h"
#include "util/Utils.h"

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , m_ui(new Ui::MainWindow)
    , m_controller(nullptr)
    , m_cloudViewer(nullptr)
{
    m_ui->setupUi(this);

    setDockNestingEnabled(true);

    m_cloudViewer = new CloudViewer(m_ui->dockWidgetContentsMainScene);
    m_ui->layoutDockMainScene->addWidget(m_cloudViewer);
//    m_cloudViewer->setCameraPosition(0, 0, 0, 0, 0, 1, 0, -1, 0);
    m_cloudViewer->setCameraPosition(0, 0, -1.5f, 0, 0, 0, 0, -1, 0);
    Eigen::Matrix4f targetPos;

    removeDockWidget(m_ui->dockWidgetMainScene);
    removeDockWidget(m_ui->dockWidgetColorImage);
    removeDockWidget(m_ui->dockWidgetDepthImage);
    removeDockWidget(m_ui->dockWidgetFrameList);
    removeDockWidget(m_ui->dockWidgetFilters);

    m_ui->dockWidgetMainScene->setVisible(true);
    m_ui->dockWidgetColorImage->setVisible(true);
    m_ui->dockWidgetDepthImage->setVisible(true);
    m_ui->dockWidgetFrameList->setVisible(true);
    m_ui->dockWidgetFilters->setVisible(true);

    addDockWidget(Qt::LeftDockWidgetArea, m_ui->dockWidgetColorImage);
    addDockWidget(Qt::RightDockWidgetArea, m_ui->dockWidgetMainScene);
    splitDockWidget(m_ui->dockWidgetColorImage, m_ui->dockWidgetDepthImage, Qt::Vertical);
    splitDockWidget(m_ui->dockWidgetMainScene, m_ui->dockWidgetFilters, Qt::Vertical);
    splitDockWidget(m_ui->dockWidgetMainScene, m_ui->dockWidgetFrameList, Qt::Horizontal);

    m_ui->dockWidgetMainScene->installEventFilter(this);
    m_ui->dockWidgetColorImage->installEventFilter(this);
    m_ui->dockWidgetDepthImage->installEventFilter(this);
    m_ui->dockWidgetFrameList->installEventFilter(this);
    m_ui->dockWidgetFilters->installEventFilter(this);

    this->setAspectRatio((720*16)/9, 720);

    m_ui->centralWidget->setVisible(false);

    connect(m_ui->actionOpen_Device, &QAction::triggered, this, &MainWindow::onActionOpenDevice);
    connect(m_ui->actionLine_Extractor, &QAction::triggered, this, &MainWindow::onActionToolWindowLineExtractor);
    connect(m_ui->actionLine_Matcher, &QAction::triggered, this, &MainWindow::onActionToolWindowLineMatcher);
    connect(m_ui->actionBoundary_Extractor, &QAction::triggered, this, &MainWindow::onActionToolWindowBoundaryExtractor);
    
    connect(m_ui->actionFused_Line_Extractor, &QAction::triggered, this, &MainWindow::onActionToolWindowFusedLineExtractor);
    connect(m_ui->actionNext_Frame, &QAction::triggered, this, &MainWindow::onActionNextFrame);
    connect(m_ui->actionStart, &QAction::triggered, this, &MainWindow::onActionStart);
    connect(m_ui->actionSave_Current_Frame, &QAction::triggered, this, &MainWindow::onActionSaveCurrentFrame);
    connect(m_ui->actionPreferences, &QAction::triggered, this, &MainWindow::onActionPreferences);
}

void MainWindow::setController(Controller *controller)
{
    if (!m_controller)
    {
        delete m_controller;
    }

    m_controller = controller;

    connect(m_controller, &Controller::frameFetched, this, &MainWindow::onFrameFetched);
    m_controller->setCloudViewer(m_cloudViewer);
}

Controller *MainWindow::controller() const
{
    return m_controller;
}

void MainWindow::setAspectRatio(int w, int h)
{
    QRect rect = this->geometry();
    if(h<100 && w<100)
    {
        // it is a ratio
        if(float(rect.width())/float(rect.height()) > float(w)/float(h))
        {
            rect.setWidth(w*(rect.height()/h));
            rect.setHeight((rect.height()/h)*h);
        }
        else
        {
            rect.setHeight(h*(rect.width()/w));
            rect.setWidth((rect.width()/w)*w);
        }
    }
    else
    {
        // it is absolute size
        rect.setWidth(w);
        rect.setHeight(h);
    }
    this->setGeometry(rect);
}

void MainWindow::onActionNextFrame()
{
    m_controller->fetchNext();
}

void MainWindow::onActionPreviousFrame(bool)
{

}

void MainWindow::onActionOpenDevice(bool checked)
{
    m_controller->open();
    if (checked)
    {
    }
}

void MainWindow::onActionStart(bool checked)
{
	m_controller->start();
}

void MainWindow::onActionPause()
{

}

void MainWindow::onActionCloseDevice()
{

}

void MainWindow::onActionToolWindowLineExtractor()
{
    m_toolWindowLineExtractor.reset(new ToolWindowLineExtractor);
    m_toolWindowLineExtractor->show();
}

void MainWindow::onActionToolWindowLineMatcher()
{
    m_toolWindowLineMatcher.reset(new ToolWindowLineMatcher);
    m_toolWindowLineMatcher->show();
}

void MainWindow::onActionToolWindowBoundaryExtractor()
{
    m_toolWindowBoundaryExtractor.reset(new ToolWindowBoundaryExtractor);
    m_toolWindowBoundaryExtractor->show();
}



void MainWindow::onActionToolWindowFusedLineExtractor()
{
    m_fusedLineExtractor.reset(new ToolWindowFusedLineExtractor);
    m_fusedLineExtractor->show();
}

void MainWindow::onActionSaveCurrentFrame()
{
    m_controller->saveCurrentFrame();
}

void MainWindow::onActionPreferences()
{
    m_preferencesWindow.reset(new PreferencesWindow);
    m_preferencesWindow->show();
}

void MainWindow::onFrameFetched(Frame &frame)
{
    //frame.showInfo();

    m_ui->widgetColorImage->setImage(frame.colorImage());
    m_ui->widgetDepthImage->setImage(frame.depthImage());

    //m_ui->dockWidgetContentsFilters->setUpdatesEnabled(false);
    //qDeleteAll(m_ui->dockWidgetContentsFilters->findChildren<ImageViewer*>("", Qt::FindDirectChildrenOnly));
    //m_ui->dockWidgetContentsFilters->setUpdatesEnabled(true);

    //for (QList<QPair<QString, cv::Mat>>::iterator i = m_controller->filteredMats().begin(); i != m_controller->filteredMats().end(); i++)
    //{
    //    QString name = i->first;
    //    cv::Mat mat = i->second;

    //    ImageViewer *widget = new ImageViewer(m_ui->dockWidgetContentsFilters);
    //    m_ui->layoutFilters->addWidget(widget);
    //    if (name.contains("depth"))
    //    {
    //        widget->setImage(cvMat2QImage(mat, false));
    //    }
    //    else if (name.contains("color"))
    //    {
    //        widget->setImage(cvMat2QImage(mat));
    //    }
    //}

    QString frameId = QString("cloud_%1").arg(frame.frameIndex());
    std::cout << "#### frame index: " << frame.deviceFrameIndex() << std::endl;
    
    //pcl::IndicesPtr indices(new std::vector<int>);
    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = frame.getCloud(*indices);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = m_controller->cloud();
    if (cloud && !cloud->empty())
    {
        //Eigen::Matrix4f pose = m_controller->pose().inverse();
        Eigen::Matrix4f pose = m_controller->pose();
        std::cout << "pose:" << std::endl;
        std::cout << pose << std::endl;

        Eigen::Matrix3f rot = pose.topLeftCorner(3, 3);
        Eigen::Vector3f rpy = rot.eulerAngles(0, 1, 2);
        const double r = ((double)rpy(0));
        const double p = ((double)rpy(1));
        const double y = ((double)rpy(2));
        Eigen::Vector3f trans = pose.topRightCorner(3, 1);
        //trans *= 10;
        std::cout << "eulers: [" << r << ", " << p << ", " << y << "], [" << trans.x() << ", " << trans.y() << ", " << trans.z() << "]" << std::endl;
        
        //if (frame.deviceFrameIndex() % 10 == 0)
        {
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmpCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
			pcl::Indices indices;
			pcl::removeNaNFromPointCloud(*cloud, *cloud, indices);
            pcl::transformPointCloud(*cloud, *tmpCloud, pose);
            //m_cloudViewer->addCloud(frameId, tmpCloud);
			pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> h(tmpCloud);
			m_cloudViewer->visualizer()->addPointCloud(tmpCloud, h, frameId.toStdString());
			m_cloudViewer->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, frameId.toStdString());
        }
    }

    QMap<qint64, Eigen::Matrix4f> poses = m_controller->poses();
    for (QMap<qint64, Eigen::Matrix4f>::iterator i = poses.begin(); i != poses.end(); i++)
    {
        qint64 index = i.key();
        Eigen::Matrix4f pose = i.value();
        frameId = QString("cloud_%1").arg(index);
        m_cloudViewer->updateCloudPose(frameId, pose);
    }

    m_cloudViewer->update();
}

MainWindow::~MainWindow()
{
}
