#include "linematchcudaodometry.h"
#include "util/StopWatch.h"
#include "cuda/CudaInternal.h"
#include "device/Device.h"
#include "ui/CloudViewer.h"
#include "extractor/LineExtractor.h"
#include "common/Parameters.h"

#include <QDebug>

#include <pcl/common/common.h>
#include <pcl/features/boundary.h>
#include <pcl/features/impl/normal_3d.hpp>
#include <pcl/features/normal_3d.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/filters/filter.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/search/kdtree.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/visualization/cloud_viewer.h>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/cudaimgproc.hpp>

bool LineMatchCudaOdometry::beforeProcessing(Frame& frame)
{
    if (!m_init)
    {
        m_init = true;
        m_colorBuffer.create(frame.getColorHeight(), frame.getColorWidth());
        m_depthBuffer.create(frame.getDepthHeight(), frame.getDepthWidth());
        m_pointCloudGpu.create(frame.getDepthWidth() * frame.getDepthHeight());
        m_pointCloudNormalsGpu.create(frame.getDepthWidth() * frame.getDepthHeight());

        m_colorMatGpu = cv::cuda::GpuMat(frame.getColorHeight(), frame.getColorWidth(), CV_8UC3, m_colorBuffer);
        m_depthMatGpu = cv::cuda::GpuMat(frame.getDepthHeight(), frame.getDepthWidth(), CV_16U, m_depthBuffer);

        m_parameters.cx = frame.getDevice()->cx();
        m_parameters.cy = frame.getDevice()->cy();
        m_parameters.fx = frame.getDevice()->fx();
        m_parameters.fy = frame.getDevice()->fy();
        m_parameters.colorWidth = frame.getColorWidth();
        m_parameters.colorHeight = frame.getColorHeight();
        m_parameters.depthWidth = frame.getDepthWidth();
        m_parameters.depthHeight = frame.getDepthHeight();
        m_parameters.depthShift = frame.getDevice()->depthShift();

        m_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
        m_normals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>());
        m_cloudIndices.reset(new std::vector<int>);

        qDebug().noquote().nospace()
            << "[LineMatchCudaOdometry::beforeProcessing] "
            << "bilateralKernelSize = " << m_bilateralFilterKernelSize
            << ", bilateralSigmaColor = " << m_bilateralFilterSigmaColor
            << ", bilateralSigmaSpatial = " << m_bilateralFilterSigmaSpatial;
    }
    return true;
}

void LineMatchCudaOdometry::doProcessing(Frame& frame)
{
    m_cloud->clear();
    m_normals->clear();
    m_cloudIndices->clear();
    TICK("odometry_uploading");
    m_colorMatGpu.upload(frame.colorMat());
    m_depthMatGpu.upload(frame.depthMat());
    TOCK("odometry_uploading");

    TICK("odometry_bilateral_filter");
    cv::cuda::bilateralFilter(m_depthMatGpu, m_depthMatGpu, m_bilateralFilterKernelSize, m_bilateralFilterSigmaColor, m_bilateralFilterSigmaSpatial);
    //cv::Ptr<cv::cuda::Filter> gaussianFilter = cv::cuda::createGaussianFilter(CV_16U, CV_16U, cv::Size(5, 5), 2.0, 1.0);
    //gaussianFilter->apply(m_depthMatGpu, m_depthMatGpu);
    TOCK("odometry_bilateral_filter");
    
    TICK("odometry_generate_point_cloud");
    m_frameGpu.colorImage = m_colorBuffer;
    m_frameGpu.depthImage = m_depthBuffer;
    m_frameGpu.pointCloud = m_pointCloudGpu;
    m_frameGpu.pointCloudNormals = m_pointCloudNormalsGpu;
    cuda::generatePointCloud(m_parameters, m_frameGpu);
    TOCK("odometry_generate_point_cloud");

    TICK("odometry_downloading");
    cv::Mat depthMatCpu;
    m_depthMatGpu.download(depthMatCpu);
    std::vector<float3> points;
    m_frameGpu.pointCloud.download(points);
    std::vector<float3> normals;
    m_frameGpu.pointCloudNormals.download(normals);

    for(int i = 0; i < frame.getDepthHeight(); i++) {
        for(int j = 0; j < frame.getDepthWidth(); j++) {
            int index = i * frame.getDepthWidth() + j;
            float3 value = points[index];
            pcl::PointXYZRGB pt;
            pcl::Normal normal;

            pt.x = value.x;
            pt.y = value.y;
            pt.z = value.z;
            pt.b = frame.colorMat().at<cv::Vec3b>(cv::Point(j, i))[0];
            pt.g = frame.colorMat().at<cv::Vec3b>(cv::Point(j, i))[1];
            pt.r = frame.colorMat().at<cv::Vec3b>(cv::Point(j, i))[2];
            normal.normal_x = normals[index].x;
            normal.normal_y = normals[index].y;
            normal.normal_z = normals[index].z;

            /*if (i == j)
            {
                qDebug() << normal.normal_x << normal.normal_y << normal.normal_z;
            }*/

            if (pt.z > 0.4f && pt.z <= 8.0f) {
            //if (qIsNaN(pt.x) || qIsNaN(pt.y) || qIsNaN(pt.y)) {
                m_cloudIndices->push_back(index);
                //qDebug() << pt.x << pt.y << pt.z;
                m_cloud->push_back(pt);
                m_normals->push_back(normal);
            }
            //else
            //{
                //pt.x = qQNaN();
                //pt.y = qQNaN();
                //pt.z = qQNaN();
            //}

            //m_cloud->push_back(pt);
            //m_normals->push_back(normal);
        }
    }

    //m_cloud->width = frame.depthMat().cols;
    //m_cloud->height = frame.depthMat().rows;
    //m_normals->width = frame.depthMat().cols;
    //m_normals->height = frame.depthMat().rows;
    TOCK("odometry_downloading");

    qDebug().nospace().noquote()
        << "[LineMatchCudaOdometry::doProcessing] "
        << "cloud indices size: " << m_cloudIndices->size();

    cv::Mat colorMat = frame.colorMat();
    cv::Mat rectifiedColorMat = frame.undistortRGBImage();
    m_filteredMats.append(QPair<QString, cv::Mat>("rectified color image", rectifiedColorMat.clone()));

    m_filteredMats.append(QPair<QString, cv::Mat>("bilateral depth image", depthMatCpu.clone()));
    cv::Mat diff = frame.depthMat() - depthMatCpu;
    m_filteredMats.append(QPair<QString, cv::Mat>("diff depth image", diff.clone()));

    //m_cloudViewer->visualizer()->addPointCloudNormals<pcl::PointXYZRGB, pcl::Normal>(m_cloud, m_normals, 100, 0.03f, "normals");

    TICK("boundary_estimation");
    // boundary estimation and extract lines
    pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud = pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr(new pcl::PointCloud<pcl::PointXYZRGBNormal>());
    pcl::concatenateFields<pcl::PointXYZRGB, pcl::Normal, pcl::PointXYZRGBNormal>(*m_cloud, *m_normals, *cloud);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloudXYZ(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud(*m_cloud, *cloudXYZ);
    pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGBNormal>());
    tree->setInputCloud(cloud);

     //calculate boundary_;
    pcl::PointCloud<pcl::Boundary> boundary;
    pcl::BoundaryEstimation<pcl::PointXYZRGBNormal, pcl::Normal, pcl::Boundary> be;
    be.setInputCloud(cloud);
    //be.setIndices(m_cloudIndices);
    be.setInputNormals(m_normals);
    be.setRadiusSearch(0.02);
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

    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> redColor(boundaryCloud, 255, 0, 0);
    //m_cloudViewer->visualizer()->addPointCloud(boundaryCloud, redColor, "boundary points");
    //m_cloudViewer->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "boundary points");
    StopWatch::instance().tock("boundary_estimation");

    StopWatch::instance().tick("line_extract");
    LineExtractor<pcl::PointXYZ, pcl::PointXYZI> le(
        PARAMETERS.floatValue("segment_distance_threshold", 0.1f, "LineExtractor"),
        PARAMETERS.intValue("min_line_points", 9, "LineExtractor"),
        PARAMETERS.floatValue("pca_error_threshold", 0.005f, "LineExtractor"),
        PARAMETERS.floatValue("line_cluster_angle_threshold", 20.0f, "LineExtractor"),
        PARAMETERS.floatValue("lines_distance_threshold", 0.01f, "LineExtractor"),
        PARAMETERS.floatValue("lines_chain_distance_threshold", 0.01f, "LineExtractor")
    );
    pcl::PointCloud<pcl::PointXYZI> leCloud;
    le.compute(*boundaryCloud, leCloud);

    pcl::PointCloud<pcl::PointXYZI>::Ptr result;
    result = le.getBoundary();

    qDebug() << "boundary size:" << result->size();
    result->width = result->points.size();
    result->height = 1;
    result->is_dense = true;
    StopWatch::instance().tock("line_extract");

    StopWatch::instance().tick("show_result");
    std::vector<LineSegment> lines = le.getLines();
    qDebug() << "size of lines: " << lines.size();

    m_cloudViewer->visualizer()->removeAllShapes();
    srand(0);
    for (size_t i = 0; i < lines.size(); i++)
    {
        std::string lineNo = "line_" + std::to_string(i);

        double r = rand() * 1.0 / RAND_MAX;
        double g = rand() * 1.0 / RAND_MAX;
        double b = rand() * 1.0 / RAND_MAX;
        //double r = 1;
        //double g = 0;
        //double b = 0;

//        qDebug().noquote().nospace() << QString::fromStdString(lineNo) << " length is " << lines[i].length() << "m";

        if (lines[i].length() > 0.f)
        {
            pcl::PointXYZI start, end;
            start.getVector3fMap() = lines[i].start();
            end.getVector3fMap() = lines[i].end();
            Eigen::Vector3f dir = lines[i].direction();
//            m_cloudViewer->visualizer()->addArrow(end, start, r, g, b, 0, lineNo);
            m_cloudViewer->visualizer()->addLine(start, end, r, g, b, lineNo);
            m_cloudViewer->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, lineNo);
        }
    }

    QList<LineCluster*> clusters = le.getLineClusters();
    for (int i = 0; i < clusters.size() && true; i++)
    {
        double r = rand() * 1.0 / RAND_MAX;
        double g = rand() * 1.0 / RAND_MAX;
        double b = rand() * 1.0 / RAND_MAX;

        LineSegment line = clusters[i]->merge();
        if (line.length() < 0.1f)
            continue;
        std::string lineNo = "cluster_line_" + std::to_string(i);
        pcl::PointXYZI start, end;
        start.getVector3fMap() = line.start();
        end.getVector3fMap() = line.end();
        m_cloudViewer->visualizer()->addLine(start, end, r, g, b, lineNo);
        m_cloudViewer->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, lineNo);

        for (int j = 0; clusters[i]->size() > 1 && j < clusters[i]->size(); j++)
        {
            std::string lineNo = "cluster_line_" + std::to_string(i) + "_" + std::to_string(j);
            pcl::PointXYZI start, end;
            start.getVector3fMap() = clusters[i]->lines()[j].start();
            end.getVector3fMap() = clusters[i]->lines()[j].end();
    //        m_cloudViewer->visualizer()->addArrow(end, start, r, g, b, 0, lineNo);
            m_cloudViewer->visualizer()->addLine(start, end, r, g, b, lineNo);
            m_cloudViewer->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 4, lineNo);
        }
    }
    StopWatch::instance().tock("show_result");

}

void LineMatchCudaOdometry::afterProcessing(Frame& frame)
{
}
