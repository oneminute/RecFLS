#include "FrameStepController.h"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <pcl/common/common.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/impl/normal_3d.hpp>
#include <pcl/features/boundary.h>
#include <pcl/visualization/cloud_viewer.h>

#include "extractor/EDLine3DExtractor.hpp"
#include "ui/CloudViewer.h"

#include <QDebug>

FrameStepController::FrameStepController(Device *device, QObject *parent)
    : Controller(device, parent)
{
    connect(m_device, &Device::frameFetched, this, &FrameStepController::onFrameFetched);
}

QString FrameStepController::name() const
{
    return "FrameStepController";
}

bool FrameStepController::open()
{
    return m_device->open();
}

void FrameStepController::close()
{
    m_device->close();
}

void FrameStepController::fetchNext()
{
    m_device->fetchNext();
}

void FrameStepController::moveTo(int frameIndex)
{
}

void FrameStepController::skip(int frameNumbers)
{
}

void FrameStepController::reset()
{
}

Frame FrameStepController::getFrame(int frameIndex)
{
    Frame frame;
    return frame;
}

void FrameStepController::onFrameFetched(Frame &frame)
{
    m_filteredMats.clear();

    // rectify image
    cv::Mat colorMat = frame.colorMat();
    cv::Mat rectifiedColorMat = m_device->undistortImage(colorMat);
    m_filteredMats.append(QPair<QString, cv::Mat>("rectified color image", rectifiedColorMat.clone()));
//    cv::imwrite("00.png", frame.colorMat());
//    cv::imwrite("01.png", rectifiedColorMat);
//    cv::Mat diff = rectifiedColorMat - frame.colorMat();
//    cv::cvtColor(diff, diff, cv::COLOR_RGB2GRAY);
//    qDebug() << cv::countNonZero(diff);

    // align depth image to color image

    // gaussian filter
    cv::Mat depthMat = frame.depthMat();
    cv::Mat tmpDepthMat;
//    cv::GaussianBlur(depthMat, tmpDepthMat, cv::Size(5, 5), 2.0, 1.0);
//    depthMat = tmpDepthMat;
//    m_filteredMats.append(QPair<QString, cv::Mat>("gaussian depth image", depthMat.clone()));

    // bilateral filter
    depthMat.convertTo(depthMat, CV_32F);
    tmpDepthMat = depthMat.clone();
    cv::bilateralFilter(depthMat, tmpDepthMat, 10, 100, 100);
//    depthMat = tmpDepthMat;
    m_filteredMats.append(QPair<QString, cv::Mat>("bilateral depth image", tmpDepthMat.clone()));
    cv::Mat diff = depthMat - tmpDepthMat;
    m_filteredMats.append(QPair<QString, cv::Mat>("diff depth image", diff.clone()));
    depthMat = tmpDepthMat;

    // generate organized point cloud
    m_cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    m_cloudIndices.reset(new std::vector<int>);

    for(int i = 0; i < depthMat.rows; i++) {
        for(int j = 0; j < depthMat.cols; j++) {
            float zValue = depthMat.at<float>(i, j);

            pcl::PointXYZRGBNormal pt;
            float Xw = 0, Yw = 0, Zw = 0;
            if(zValue > 0 && !qIsNaN(zValue)) {
                Zw = zValue / m_device->depthShift();
                Xw = (j - m_device->cx()) *  Zw / m_device->fx();
                Yw = (i - m_device->cy()) * Zw / m_device->fy();
                m_cloudIndices->push_back(i * depthMat.cols + j);
            }

            pt.x = Xw;
            pt.y = Yw;
            pt.z = Zw;
            pt.b = colorMat.at<cv::Vec3b>(cv::Point(j, i))[0];
            pt.g = colorMat.at<cv::Vec3b>(cv::Point(j, i))[1];
            pt.r = colorMat.at<cv::Vec3b>(cv::Point(j, i))[2];

            m_cloud->push_back(pt);
        }
    }
    m_cloud->resize(m_cloud->points.size());
    m_cloud->width = static_cast<quint32>(colorMat.cols);
    m_cloud->height = static_cast<quint32>(colorMat.rows);

    // boundary estimation and extract lines
    pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBNormal>());
    tree->setInputCloud(m_cloud);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZRGBNormal, pcl::Normal> normal_est;
    normal_est.setSearchMethod(tree);
    normal_est.setInputCloud(m_cloud);
    normal_est.setKSearch(5);
    normal_est.compute(*normals);

    //calculate boundary_;
    pcl::PointCloud<pcl::Boundary> boundary;
    pcl::EDLine3DExtractor<pcl::PointXYZRGBNormal, pcl::Normal, pcl::Boundary> edline3dExtractor;
    edline3dExtractor.setInputCloud(m_cloud);
    edline3dExtractor.setInputNormals(normals);
    edline3dExtractor.setRadiusSearch(0.01);
    //edline3dExtractor.setAngleThreshold(PI/4);
    edline3dExtractor.setSearchMethod(tree);
    edline3dExtractor.compute(boundary);

    m_result = edline3dExtractor.getBoundary();

    qDebug() << "boundary size:" << m_result->size();
    m_result->width = m_result->points.size();
    m_result->height = 1;
    m_result->is_dense = true;

    std::vector<pcl::EDLine3D> lines = edline3dExtractor.getLines();
    std::vector<pcl::EDLine3D> mergedLines = edline3dExtractor.getMergedLines();
    qDebug() << "size of lines: " << lines.size() << ", size of merged lines: " << mergedLines.size();

    m_cloudViewer->visualizer()->removeAllShapes();
    srand(0);
    for (int i = 0; i < mergedLines.size(); i++)
    {
        std::string lineNo = "merged_line_" + std::to_string(i);

        double r = rand() * 1.0 / RAND_MAX;
        double g = rand() * 1.0 / RAND_MAX;
        double b = rand() * 1.0 / RAND_MAX;

        qDebug() << QString::fromStdString(lineNo) << r << g << b;

        pcl::PointXYZI start = mergedLines[i].start;
        pcl::PointXYZI end = mergedLines[i].end;
        m_cloudViewer->visualizer()->addLine(start, end, r, g, b, lineNo);
        m_cloudViewer->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 4, lineNo);
    }

//    m_cloudViewer->visualizer()->removeAllShapes();
//    if (m_lastMergedLines.size() > 0)
//    {
//        std::map<int, int> pairs = edline3dExtractor.linesCompare2(m_lastMergedLines);
//        std::cout << "pairs: " << pairs.size() << std::endl;
//        srand(0);
//        for (int i = 0; i < mergedLines.size(); i++)
//        {
//            if (pairs.find(i) == pairs.end())
//                continue;
//            int targetLineIndex = pairs[i];
//            if (targetLineIndex >= 0)
//            {
//                std::cout << i << " --> " << targetLineIndex << std::endl;
//                std::string lineNo = "merged_line_" + std::to_string(i);
//                std::string targetLineNo = "merged_line_target_" + std::to_string(i);

//                std::string line1 = "rel_line_1_" + std::to_string(i);
//                std::string line2 = "rel_line_2_" + std::to_string(i);

//                double r = rand() * 1.0 / RAND_MAX;
//                double g = rand() * 1.0 / RAND_MAX;
//                double b = rand() * 1.0 / RAND_MAX;

//                m_cloudViewer->visualizer()->addLine(mergedLines[targetLineIndex].start, mergedLines[targetLineIndex].end, r, g, b, targetLineNo);
//                m_cloudViewer->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 4, targetLineNo);

//                pcl::PointXYZI start = m_lastMergedLines[i].start;
//                pcl::PointXYZI end = m_lastMergedLines[i].end;
//                start.x -= 2;
//                end.x -= 2;
//                m_cloudViewer->visualizer()->addLine(start, end, r, b, b, lineNo);
//                m_cloudViewer->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 4, lineNo);

//                m_cloudViewer->visualizer()->addLine(start, mergedLines[targetLineIndex].start, r, g, b, line1);
//                m_cloudViewer->visualizer()->addLine(end, mergedLines[targetLineIndex].end, r, g, b, line2);
//                m_cloudViewer->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, line1);
//                m_cloudViewer->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 1, line2);
//            }
//        }
//    }

    m_lastMergedLines = mergedLines;
    m_lastLineCloud = edline3dExtractor.getLineCloud();

//    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> redColor(result, 255, 0, 0);
//    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> iColor(result, "intensity");
//    m_cloudViewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "boundary points");

    // generate line descriptors

    // match

    // calculate transformation

    // emit signal
    emit frameFetched(frame);
}
