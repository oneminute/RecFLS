#include "FrameStepController.h"
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <pcl/common/common.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/organized.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/impl/normal_3d.hpp>
#include <pcl/features/boundary.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/range_image/range_image.h>
#include <pcl/range_image/range_image_planar.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/filters/filter.h>

//#include "extractor/EDLine3DExtractor.hpp"
#include "extractor/LineExtractor.hpp"
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
            Zw = zValue / m_device->depthShift();
            Xw = (j - m_device->cx()) *  Zw / m_device->fx();
            Yw = (i - m_device->cy()) * Zw / m_device->fy();

            pt.x = Xw;
            pt.y = Yw;
            pt.z = Zw;
            pt.b = colorMat.at<cv::Vec3b>(cv::Point(j, i))[0];
            pt.g = colorMat.at<cv::Vec3b>(cv::Point(j, i))[1];
            pt.r = colorMat.at<cv::Vec3b>(cv::Point(j, i))[2];

//            if(zValue > 0 && !qIsNaN(zValue)) {
            if (pt.getVector3fMap().norm() > 0.1f) {
                m_cloudIndices->push_back(i * depthMat.cols + j);
            }
            else
            {
                Xw = qQNaN();
                Yw = qQNaN();
                Zw = qQNaN();
                continue;
            }


//            if (pt.getVector3fMap().norm() > 0)
            m_cloud->push_back(pt);
        }
    }
//    m_cloud->resize(m_cloud->points.size());
//    m_cloud->width = static_cast<quint32>(colorMat.cols);
//    m_cloud->height = static_cast<quint32>(colorMat.rows);
    qDebug() << "isOrganized:" << m_cloud->isOrganized() << m_cloudIndices->size();
    m_cloudViewer->visualizer()->removeAllShapes();

    float angular_resolution = 0.001f;
    float support_size = 0.2f;
    pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
    bool setUnseenToMaxRange = false;

    // ------------------------------------------------------------------
    // -----Read pcd file or create example point cloud if not given-----
    // ------------------------------------------------------------------
    pcl::PointCloud<pcl::PointWithViewpoint> far_ranges;
    Eigen::Affine3f scene_sensor_pose(Eigen::Affine3f::Identity());
    // -----Create RangeImage from the PointCloud-----
    // -----------------------------------------------
    float noise_level = 0.0;
    float min_range = 0.0f;
    int border_size = 1;
    pcl::RangeImage::Ptr range_image_ptr(new pcl::RangeImage);
    pcl::RangeImage& range_image = *range_image_ptr;

    std::vector<int> mapping;
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr tmpCloud(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    pcl::removeNaNFromPointCloud(*m_cloud, *tmpCloud, mapping);

    range_image.createFromPointCloud(*tmpCloud, angular_resolution, pcl::deg2rad(360.0f), pcl::deg2rad(360.0f),
        scene_sensor_pose, coordinate_frame, noise_level, min_range, border_size);
    range_image.integrateFarRanges(far_ranges);
    if (setUnseenToMaxRange)
        range_image.setUnseenToMaxRange();
    qDebug() << "width:" << range_image.width << ", height:" << range_image.height;

    //calculate RangeImageBorder;
    pcl::RangeImageBorderExtractor border_extractor(&range_image);//定义边界提取对对象，形参为一个RangeImage类型的指针
    border_extractor.setRadiusSearch(0.01);
    pcl::PointCloud<pcl::BorderDescription> border_descriptions;
    border_extractor.compute(border_descriptions);//将边界信息存放到board_description

    pcl::PointCloud<pcl::PointWithRange>::Ptr border_points_ptr(new pcl::PointCloud<pcl::PointWithRange>),
            veil_points_ptr(new pcl::PointCloud<pcl::PointWithRange>),
            shadow_points_ptr(new pcl::PointCloud<pcl::PointWithRange>);
    pcl::PointCloud<pcl::PointWithRange>& border_points = *border_points_ptr, & veil_points = * veil_points_ptr, & shadow_points = *shadow_points_ptr;//创建指针，类模板是PointCloud<pcl::PointCloud>类
    for (int y=0; y< (int)range_image.height; ++y)
    {
        for (int x=0; x< (int)range_image.width; ++x)//遍历深度图像中的每一个点
        {
            if (border_descriptions.points[y*range_image.width + x].traits[pcl::BORDER_TRAIT__OBSTACLE_BORDER])
                border_points.points.push_back (range_image.points[y*range_image.width + x]);
            if (border_descriptions.points[y*range_image.width + x].traits[pcl::BORDER_TRAIT__VEIL_POINT])
                veil_points.points.push_back (range_image.points[y*range_image.width + x]);
            if (border_descriptions.points[y*range_image.width + x].traits[pcl::BORDER_TRAIT__SHADOW_BORDER])
                shadow_points.points.push_back (range_image.points[y*range_image.width + x]);
        }
    }
    qDebug() << "boundary size:" << border_points.size() << ", veil size:" << veil_points.size() << ", shadow size:" << shadow_points.size() << ", cloud size:" << m_cloud->points.size();
    //可视化三组边界到statical滤波之后的点云
//    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithRange> veil_points_color_handler (veil_points_ptr, 0, 255, 0);
//    m_cloudViewer->visualizer()->addPointCloud<pcl::PointWithRange> (veil_points_ptr, veil_points_color_handler, "veil points");
//    m_cloudViewer->visualizer()->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "veil points");
//    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithRange> border_points_color_handler (border_points_ptr, 255, 0, 0);
//    m_cloudViewer->visualizer()->addPointCloud<pcl::PointWithRange> (border_points_ptr, border_points_color_handler, "border points");
//    m_cloudViewer->visualizer()->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "border points");
//    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointWithRange> shadow_points_color_handler (shadow_points_ptr, 0, 0, 255);
//    m_cloudViewer->visualizer()->addPointCloud<pcl::PointWithRange> (shadow_points_ptr, shadow_points_color_handler, "shadow points");
//    m_cloudViewer->visualizer()->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "shadow points");

    LineExtractor<pcl::PointWithRange, pcl::PointXYZI> le;
    pcl::PointCloud<pcl::PointXYZI> leCloud;
    le.compute(border_points, leCloud);

//    m_cloudViewer->visualizer()->addPointCloud(cloud_temp, keypoints_color_handler, "keypoints");
//    m_cloudViewer->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 4, "keypoints");

//    // boundary estimation and extract lines
//    pcl::search::OrganizedNeighbor<pcl::PointXYZRGBNormal>::Ptr tree(new pcl::search::OrganizedNeighbor<pcl::PointXYZRGBNormal>());
//    tree->setInputCloud(m_cloud);
//    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
//    pcl::NormalEstimation<pcl::PointXYZRGBNormal, pcl::Normal> normal_est;
//    normal_est.setSearchMethod(tree);
//    normal_est.setInputCloud(m_cloud);
//    normal_est.setKSearch(5);
//    normal_est.compute(*normals);

//    //calculate boundary_;
//    pcl::PointCloud<pcl::Boundary> boundary;
//    pcl::EDLine3DExtractor<pcl::PointXYZRGBNormal, pcl::Normal, pcl::Boundary> edline3dExtractor;
//    edline3dExtractor.setInputCloud(m_cloud);
//    edline3dExtractor.setInputNormals(normals);
//    edline3dExtractor.setRadiusSearch(0.01);
//    //edline3dExtractor.setAngleThreshold(PI/4);
//    edline3dExtractor.setSearchMethod(tree);
//    edline3dExtractor.compute(boundary);

    m_result = le.getBoundary();

    qDebug() << "boundary size:" << m_result->size();
    m_result->width = m_result->points.size();
    m_result->height = 1;
    m_result->is_dense = true;

    std::vector<LineSegment> lines = le.getLines();
    std::vector<LineSegment> mergedLines = le.getMergedLines();
    qDebug() << "size of lines: " << lines.size() << ", size of merged lines: " << mergedLines.size();

    m_cloudViewer->visualizer()->removeAllShapes();
    srand(0);
    for (int i = 0; i < lines.size(); i++)
    {
        std::string lineNo = "merged_line_" + std::to_string(i);

        double r = rand() * 1.0 / RAND_MAX;
        double g = rand() * 1.0 / RAND_MAX;
        double b = rand() * 1.0 / RAND_MAX;

        qDebug() << QString::fromStdString(lineNo) << r << g << b;

        pcl::PointXYZI start, end;
        start.getVector3fMap() = lines[i].start();
        end.getVector3fMap() = lines[i].end();
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

//    m_lastMergedLines = mergedLines;
//    m_lastLineCloud = edline3dExtractor.getLineCloud();

//    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> redColor(result, 255, 0, 0);
//    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> iColor(result, "intensity");
//    m_cloudViewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "boundary points");

    // generate line descriptors

    // match

    // calculate transformation

    // emit signal
    emit frameFetched(frame);
}
