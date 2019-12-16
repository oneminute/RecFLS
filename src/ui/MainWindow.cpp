#include "MainWindow.h"
#include "ui_MainWindow.h"
#include "util/Utils.h"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>

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
    connect(m_ui->actionNext_Frame, &QAction::triggered, this, &MainWindow::onActionNextFrame);
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

    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud = m_controller->cloud();
//    std::vector<int> mapping;
//    pcl::removeNaNFromPointCloud(*cloud, *cloud, mapping);
    m_cloudViewer->addCloud("main scene", cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr result(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*m_controller->result(), *result);
    m_cloudViewer->addCloud("boundary points", result, Eigen::Matrix4f::Identity(), QColor(255, 0, 0, 255));
}

MainWindow::~MainWindow()
{
}
