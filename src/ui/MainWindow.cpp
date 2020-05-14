#include "MainWindow.h"
#include "ui_MainWindow.h"
#include "util/Utils.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/common/pca.h>

#include "ToolWindowLineExtractor.h"
#include "ToolWindowLineMatcher.h"
#include "ToolWindowBoundaryExtractor.h"
#include "ToolWindowICPMatcher.h"
#include "ToolWindowFusedLineExtractor.h"
#include "PreferencesWindow.h"

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
    connect(m_ui->actionICP_Matcher, &QAction::triggered, this, &MainWindow::onActionToolWindowICPMatcher);
    connect(m_ui->actionFused_Line_Extractor, &QAction::triggered, this, &MainWindow::onActionToolWindowFusedLineExtractor);
    connect(m_ui->actionNext_Frame, &QAction::triggered, this, &MainWindow::onActionNextFrame);
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

void MainWindow::onActionToolWindowICPMatcher()
{
    m_toolWindowICPMatcher.reset(new ToolWindowICPMatcher);
    m_toolWindowICPMatcher->show();
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
    frame.showInfo();

    m_ui->widgetColorImage->setImage(cvMat2QImage(frame.colorMat()));
    m_ui->widgetDepthImage->setImage(cvMat2QImage(frame.depthMat(), false));

    m_ui->dockWidgetContentsFilters->setUpdatesEnabled(false);
    qDeleteAll(m_ui->dockWidgetContentsFilters->findChildren<ImageViewer*>("", Qt::FindDirectChildrenOnly));
    m_ui->dockWidgetContentsFilters->setUpdatesEnabled(true);

    for (QList<QPair<QString, cv::Mat>>::iterator i = m_controller->filteredMats().begin(); i != m_controller->filteredMats().end(); i++)
    {
        QString name = i->first;
        cv::Mat mat = i->second;

        ImageViewer *widget = new ImageViewer(m_ui->dockWidgetContentsFilters);
        m_ui->layoutFilters->addWidget(widget);
        if (name.contains("depth"))
        {
            widget->setImage(cvMat2QImage(mat, false));
        }
        else if (name.contains("color"))
        {
            widget->setImage(cvMat2QImage(mat));
        }
    }

    QString frameId = QString("cloud_%1").arg(frame.deviceFrameIndex());
    std::cout << "#### frame index: " << frame.deviceFrameIndex() << std::endl;
    
    pcl::IndicesPtr indices(new std::vector<int>);
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = frame.getCloud(*indices);
    //pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud = m_controller->cloud();
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
        pose.topRightCorner(3, 1) = trans;
        std::cout << "eulers: [" << r << ", " << p << ", " << y << "], [" << trans.x() << ", " << trans.y() << ", " << trans.z() << "]" << std::endl;
        
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmpCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        //Eigen::Matrix4f tmp(Eigen::Matrix4f::Identity());
        //tmp.topRightCorner(3, 1) += Eigen::Vector3f(0.1f * frame.deviceFrameIndex(), 0, 0);
        pcl::transformPointCloud(*cloud, *tmpCloud, pose);
        //m_cloudViewer->removeAllClouds();
        m_cloudViewer->addCloud(frameId, tmpCloud);
    }

    m_cloudViewer->update();
}

MainWindow::~MainWindow()
{
}
