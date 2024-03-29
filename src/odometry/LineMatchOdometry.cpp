#include "LineMatchOdometry.h"
#include "device/Device.h"
#include "util/StopWatch.h"

#include <QDebug>

#include <pcl/common/common.h>
#include <pcl/features/boundary.h>
#include <pcl/features/impl/normal_3d.hpp>
#include <pcl/features/normal_3d.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/filters/filter.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/organized.h>
#include <pcl/range_image/range_image.h>
#include <pcl/range_image/range_image_planar.h>
#include <pcl/visualization/cloud_viewer.h>

#include <pcl/gpu/features/features.hpp>

LineMatchOdometry::LineMatchOdometry(QObject* parent)
    : Odometry(nullptr)
{

}

bool LineMatchOdometry::beforeProcessing(Frame& frame)
{
    return true;
}

void LineMatchOdometry::doProcessing(Frame& frame)
{
    StopWatch::instance().tick("frame_processing");
    m_filteredMats.clear();

    // rectify image
    StopWatch::instance().tick("image_filter");
    cv::Mat colorMat = frame.colorMat();
    cv::Mat rectifiedColorMat = frame.undistortRGBImage();
    m_filteredMats.append(QPair<QString, cv::Mat>("rectified color image", rectifiedColorMat.clone()));

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
    StopWatch::instance().tock("image_filter");

    StopWatch::instance().tick("generate_cloud");
    // generate organized point cloud
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    m_cloudIndices.reset(new std::vector<int>);

    for(int i = 0; i < depthMat.rows; i++) {
        for(int j = 0; j < depthMat.cols; j++) {
            float zValue = depthMat.at<float>(i, j);

            pcl::PointXYZRGBNormal pt;
            float Xw = 0, Yw = 0, Zw = 0;
            Zw = zValue / frame.getDevice()->depthShift();
            Xw = (j - frame.getDevice()->cx()) * Zw / frame.getDevice()->fx();
            Yw = (i - frame.getDevice()->cy()) * Zw / frame.getDevice()->fy();

            pt.x = Xw;
            pt.y = Yw;
            pt.z = Zw;
            pt.b = colorMat.at<cv::Vec3b>(cv::Point(j, i))[0];
            pt.g = colorMat.at<cv::Vec3b>(cv::Point(j, i))[1];
            pt.r = colorMat.at<cv::Vec3b>(cv::Point(j, i))[2];

//            if(zValue > 0 && !qIsNaN(zValue)) {
            bool added = false;
            if (pt.getVector3fMap().norm() > 0.1f) {
                added = true;
                m_cloudIndices->push_back(i * depthMat.cols + j);
            }
            else
            {
                Xw = qQNaN();
                Yw = qQNaN();
                Zw = qQNaN();
                continue;
            }

            //qDebug() << QString("[%1,%2]").arg(i).arg(j) << pt.x << pt.y << pt.z;

//            if (pt.getVector3fMap().norm() > 0)
            cloud->push_back(pt);
        }
    }
    StopWatch::instance().tock("generate_cloud");
//    m_cloud->resize(m_cloud->points.size());
//    m_cloud->width = static_cast<quint32>(colorMat.cols);
//    m_cloud->height = static_cast<quint32>(colorMat.rows);
    pcl::copyPointCloud(*cloud, *m_cloud);

    StopWatch::instance().tick("normal_estimation");
    qDebug() << "isOrganized:" << m_cloud->isOrganized() << m_cloudIndices->size();
    m_cloudViewer->visualizer()->removeAllShapes();

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudXYZ(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*m_cloud, *cloudXYZ);

    pcl::gpu::NormalEstimation::PointCloud gpuCloud;
    pcl::gpu::NormalEstimation::Indices gpuIndices;

    gpuCloud.upload(cloudXYZ->points);
    gpuIndices.upload(m_cloudIndices->data(), m_cloudIndices->size());

    pcl::gpu::NormalEstimation::Normals gpuNormals;
    pcl::gpu::NormalEstimation ne;
    ne.setInputCloud(gpuCloud);
//    ne.setIndices(gpuIndices);
    ne.setViewPoint(0, 0, 0);
    ne.setRadiusSearch(0.01f, 50);
    ne.compute(gpuNormals);

    std::vector<pcl::PointXYZ> cpuNormals;
    gpuNormals.download(cpuNormals);

    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    for (std::vector<pcl::PointXYZ>::iterator i = cpuNormals.begin(); i != cpuNormals.end(); i++)
    {
        pcl::Normal n;
        n._Normal::normal_x = i->x;
        n._Normal::normal_y = i->y;
        n._Normal::normal_z = i->z;
        n._Normal::curvature = i->data[3];
        normals->push_back(n);
    }

    qDebug() << "normal size:" << normals->size();
    StopWatch::instance().tock("normal_estimation");

    StopWatch::instance().tick("boundary_estimation");
//    // boundary estimation and extract lines
    pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBNormal>());
    tree->setInputCloud(cloud);

     //calculate boundary_;
    pcl::PointCloud<pcl::Boundary> boundary;
    pcl::BoundaryEstimation<pcl::PointXYZRGBNormal, pcl::Normal, pcl::Boundary> be;
    be.setInputCloud(cloud);
//    be.setIndices(m_cloudIndices);
    be.setInputNormals(normals);
    be.setRadiusSearch(0.01);
    be.setAngleThreshold(M_PI_4);
    be.setSearchMethod(tree);
    be.compute(boundary);

    pcl::PointCloud<pcl::PointXYZ>::Ptr boundaryCloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (int i = 0; i < boundary.points.size(); i++)
    {
        if (boundary[i].boundary_point == 1)
        {
            boundaryCloud->points.push_back(cloudXYZ->points[i]);
        }
    }
    boundaryCloud->width = boundaryCloud->points.size();
    boundaryCloud->height = 1;
    boundaryCloud->is_dense = true;
    std::cout << "boundary size:" << boundary.size() << ", cloud size:" << m_cloud->points.size() << std::endl;
    StopWatch::instance().tock("boundary_estimation");

//    StopWatch::instance().tick("line_extract");
//    LineExtractor<pcl::PointXYZ, pcl::PointXYZI> le;
//    pcl::PointCloud<pcl::PointXYZI> leCloud;
//    le.compute(*boundaryCloud, leCloud);

//    pcl::PointCloud<pcl::PointXYZI>::Ptr result;
//    result = le.getBoundary();

//    qDebug() << "boundary size:" << result->size();
//    result->width = result->points.size();
//    result->height = 1;
//    result->is_dense = true;
//    StopWatch::instance().tock("line_extract");

//    StopWatch::instance().tick("show_result");
//    std::vector<LineSegment> lines = le.getLines();
//    qDebug() << "size of lines: " << lines.size();

//    m_cloudViewer->visualizer()->removeAllShapes();
//    srand(0);
//    for (size_t i = 0; i < lines.size() && false; i++)
//    {
//        std::string lineNo = "line_" + std::to_string(i);

//        double r = rand() * 1.0 / RAND_MAX;
//        double g = rand() * 1.0 / RAND_MAX;
//        double b = rand() * 1.0 / RAND_MAX;
////        double r = 1;
////        double g = 0;
////        double b = 0;

////        qDebug().noquote().nospace() << QString::fromStdString(lineNo) << " length is " << lines[i].length() << "m";

//        if (lines[i].length() > 0.1f)
//        {
//            pcl::PointXYZI start, end;
//            start.getVector3fMap() = lines[i].start();
//            end.getVector3fMap() = lines[i].end();
//            Eigen::Vector3f dir = lines[i].direction();
////            m_cloudViewer->visualizer()->addArrow(end, start, r, g, b, 0, lineNo);
//            m_cloudViewer->visualizer()->addLine(start, end, r, g, b, lineNo);
//            m_cloudViewer->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, lineNo);
//        }
//    }

//    QList<LineCluster*> clusters = le.getLineClusters();
//    for (int i = 0; i < clusters.size() && true; i++)
//    {
//        double r = rand() * 1.0 / RAND_MAX;
//        double g = rand() * 1.0 / RAND_MAX;
//        double b = rand() * 1.0 / RAND_MAX;

//        LineSegment line = clusters[i]->merge();
//        if (line.length() < 0.1f)
//            continue;
//        std::string lineNo = "cluster_line_" + std::to_string(i);
//        pcl::PointXYZI start, end;
//        start.getVector3fMap() = line.start();
//        end.getVector3fMap() = line.end();
//        m_cloudViewer->visualizer()->addLine(start, end, r, g, b, lineNo);
//        m_cloudViewer->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 4, lineNo);

//        for (int j = 0; clusters[i]->size() > 3 && j < clusters[i].size(); j++)
//        {
//            std::string lineNo = "cluster_line_" + std::to_string(i) + "_" + std::to_string(j);
//            pcl::PointXYZI start, end;
//            start.getVector3fMap() = clusters[i][j].start();
//            end.getVector3fMap() = clusters[i][j].end();
//    //        m_cloudViewer->visualizer()->addArrow(end, start, r, g, b, 0, lineNo);
//            m_cloudViewer->visualizer()->addLine(start, end, r, g, b, lineNo);
//            m_cloudViewer->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 4, lineNo);
//        }
//    }
    StopWatch::instance().tock("show_result");

//    pcl::PointCloud<pcl::PointXYZI>::Ptr segmentsCloud(new pcl::PointCloud<pcl::PointXYZI>);
//    std::vector<std::vector<int>> segments = le.getSegments();
//    srand(QDateTime::currentMSecsSinceEpoch());
//    for (int s = 0; s < segments.size(); s++)
//    {
//        double r = rand() * 1.0 / RAND_MAX;
//        for (int i = 0; i < segments[s].size(); i++)
//        {
//            pcl::PointXYZI pt = m_result->points[segments[s][i]];
//            pt.intensity = r;
//            segmentsCloud->push_back(pt);
//        }
//    }
//    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> iColor(segmentsCloud, "intensity");
//    m_cloudViewer->visualizer()->addPointCloud(segmentsCloud, iColor, "result");

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

    StopWatch::instance().tock("frame_processing");
}

void LineMatchOdometry::afterProcessing(Frame& frame)
{
}
