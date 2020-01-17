#include "LineMatchCudaOdometry.h"
#include "util/StopWatch.h"
#include "cuda/CudaInternal.h"
#include "device/Device.h"
#include "ui/CloudViewer.h"
//#include "extractor/LineExtractor.hpp"
#include "common/Parameters.h"

#include <QDebug>
#include <QFileDialog>

#include <pcl/common/common.h>
#include <pcl/features/boundary.h>
#include <pcl/features/impl/normal_3d.hpp>
#include <pcl/features/normal_3d.h>
#include <pcl/features/range_image_border_extractor.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/bilateral.h>
#include <pcl/filters/fast_bilateral.h>
#include <pcl/filters/convolution_3d.h>
#include <pcl/filters/convolution.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/search/kdtree.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/obj_io.h>
#include <pcl/io/pcd_io.h>

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

void LineMatchCudaOdometry::saveCurrentFrame()
{
    QString fileName = QFileDialog::getSaveFileName(nullptr, tr("Save Boundaries"), QDir::currentPath(), tr("Polygon Files (*.obj *.ply *.pcd)"));
    qDebug() << "saving file" << fileName;
    pcl::PointCloud<pcl::PointXYZ> cloud;
    pcl::copyPointCloud(*m_boundaryCloud, cloud);
    qDebug() << m_boundaryCloud->size() << cloud.size();
    cloud.width = cloud.size();
    cloud.height = 1;
    cloud.is_dense = true;
    pcl::io::savePCDFile<pcl::PointXYZ>(fileName.toStdString(), cloud);
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

    //TICK("odometry_bilateral_filter");
    //cv::cuda::bilateralFilter(m_depthMatGpu, m_depthMatGpu, m_bilateralFilterKernelSize, m_bilateralFilterSigmaColor, m_bilateralFilterSigmaSpatial);
    //cv::Ptr<cv::cuda::Filter> gaussianFilter = cv::cuda::createGaussianFilter(CV_16U, CV_16U, cv::Size(5, 5), 2.0, 1.0);
    //gaussianFilter->apply(m_depthMatGpu, m_depthMatGpu);
    //TOCK("odometry_bilateral_filter");
    
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

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmpCloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    for(int i = 0; i < frame.getDepthHeight(); i++) {
        for(int j = 0; j < frame.getDepthWidth(); j++) {
            int index = i * frame.getDepthWidth() + j;
            float3 value = points[index];
            pcl::PointXYZRGB pt;
            //pcl::Normal normal;

            pt.x = value.x;
            pt.y = value.y;
            pt.z = value.z;
            //pt.b = frame.colorMat().at<cv::Vec3b>(cv::Point(j, i))[0];
            //pt.g = frame.colorMat().at<cv::Vec3b>(cv::Point(j, i))[1];
            //pt.r = frame.colorMat().at<cv::Vec3b>(cv::Point(j, i))[2];
            //pt.b = 0;
            //pt.g = 127;
            //pt.r = 0;
            //normal.normal_x = normals[index].x;
            //normal.normal_y = normals[index].y;
            //normal.normal_z = normals[index].z;

            /*if (i == j)
            {
                qDebug() << normal.normal_x << normal.normal_y << normal.normal_z;
            }*/

            if (pt.z > 0.4f && pt.z <= 8.0f) {
            //if (qIsNaN(pt.x) || qIsNaN(pt.y) || qIsNaN(pt.y)) {
                m_cloudIndices->push_back(index);
                //qDebug() << pt.x << pt.y << pt.z;
                //m_normals->push_back(normal);
                tmpCloud->push_back(pt);
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
    //tmpCloud->width = frame.getDepthWidth();
    //tmpCloud->height = frame.getDepthHeight();
    tmpCloud->is_dense = true;

    pcl::PointCloud<pcl::PointXYZRGB> tmpCloud2;

    pcl::filters::GaussianKernel<pcl::PointXYZRGB, pcl::PointXYZRGB>::Ptr kernel(new pcl::filters::GaussianKernel<pcl::PointXYZRGB, pcl::PointXYZRGB>);
    kernel->setSigma(4);
    kernel->setThresholdRelativeToSigma(4);

    //Set up the Convolution Filter
    pcl::filters::Convolution3D<pcl::PointXYZRGB, pcl::PointXYZRGB, pcl::filters::GaussianKernel<pcl::PointXYZRGB, pcl::PointXYZRGB>> convolution;
    convolution.setKernel(*kernel);
    convolution.setInputCloud(tmpCloud);
    convolution.setRadiusSearch(0.1);
    convolution.convolve(tmpCloud2);
   
    /*pcl::FastBilateralFilter<pcl::PointXYZ> bf;
    bf.setInputCloud(tmpCloud);
    bf.setIndices(m_cloudIndices);
    bf.setSigmaS(50);
    bf.setSigmaR(0.1);
    bf.filter(tmpCloud2);*/

    //pcl::copyPointCloud(tmpCloud2, *m_cloudIndices, *m_cloud);
    pcl::copyPointCloud(tmpCloud2, *m_cloud);
    m_cloud->width = m_cloud->points.size();
    m_cloud->height = 1;
    //m_cloud->is_dense = true;

    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> ne;
    //ne.setIndices(m_cloudIndices);
    ne.setInputCloud(m_cloud);
    ne.setRadiusSearch(0.1f);
    ne.compute(*m_normals);

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

    //m_filteredMats.append(QPair<QString, cv::Mat>("bilateral depth image", depthMatCpu.clone()));
    //cv::Mat diff = frame.depthMat() - depthMatCpu;
    //m_filteredMats.append(QPair<QString, cv::Mat>("diff depth image", diff.clone()));

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
    be.setRadiusSearch(0.1);
    be.setAngleThreshold(M_PI_4);
    //be.setSearchMethod(tree);
    be.compute(boundary);

    m_boundaryCloud = pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>);
    for (int i = 0; i < boundary.points.size(); i++)
    {
        if (boundary[i].boundary_point == 1)
        {
            pcl::PointXYZI pt;
            pt.x = cloudXYZ->points[i].x;
            pt.y = cloudXYZ->points[i].y;
            pt.z = cloudXYZ->points[i].z;
            pt.intensity = i;
            m_boundaryCloud->points.push_back(pt);
        }
    }
    m_boundaryCloud->width = m_boundaryCloud->points.size();
    m_boundaryCloud->height = 1;
    m_boundaryCloud->is_dense = true;
    std::cout << "boundary size:" << boundary.size() << ", cloud size:" << m_cloud->points.size() << std::endl;
    StopWatch::instance().tock("boundary_estimation");

    /*StopWatch::instance().tick("line_extract");
    LineExtractor<pcl::PointXYZI, pcl::PointXYZI> le(
        PARAMETERS.floatValue("segment_distance_threshold", 0.1f, "LineExtractor"),
        PARAMETERS.intValue("min_line_points", 9, "LineExtractor"),
        PARAMETERS.floatValue("pca_error_threshold", 0.005f, "LineExtractor"),
        PARAMETERS.floatValue("line_cluster_angle_threshold", 20.0f, "LineExtractor"),
        PARAMETERS.floatValue("lines_distance_threshold", 0.01f, "LineExtractor"),
        PARAMETERS.floatValue("lines_chain_distance_threshold", 0.01f, "LineExtractor")
    );
    pcl::PointCloud<pcl::PointXYZI> leCloud;
    le.compute(*m_boundaryCloud, leCloud);

    pcl::PointCloud<pcl::PointXYZI>::Ptr result;
    result = le.getBoundary();

    qDebug() << "boundary size:" << result->size();
    result->width = result->points.size();
    result->height = 1;
    result->is_dense = true;
    StopWatch::instance().tock("line_extract");*/

    StopWatch::instance().tick("show_result");
    //std::vector<LineSegment> lines = le.getLines();
    //qDebug() << "size of lines: " << lines.size();

    //pcl::PointCloud<pcl::PointXYZ>::Ptr dirCloud;
    ////pcl::PointCloud<pcl::PointXYZI>::Ptr clusterCloud = le.parameterizedPointMappingCluster(dirCloud);
    //pcl::PointCloud<pcl::PointXYZI>::Ptr clusterCloud = le.parameterizedLineMappingCluster();
    //QList<QList<int>> indicesCluster = le.lineClusterFromParameterizedPointCloud(clusterCloud);
    //QList<LineSegment> clusterLines = le.extractLinesFromClusters(indicesCluster, clusterCloud);
    //qDebug() << "size of cluster lines: " << clusterLines.size();
    //int clusterIndex = 0;
    //srand(0);
    //m_cloudViewer->visualizer()->removeAllShapes();

    //for (QList<QList<int>>::iterator i = indicesCluster.begin(); i != indicesCluster.end() && true; i++)
    //{
    //    double r = rand() * 1.0 / RAND_MAX;
    //    double g = rand() * 1.0 / RAND_MAX;
    //    double b = rand() * 1.0 / RAND_MAX;

    //    for (QList<int>::iterator j = i->begin(); j != i->end(); j++)
    //    {
    //        int index = clusterCloud->points[*j].intensity;
    //        LineSegment line = lines[index];
    //        std::string lineNo = "line_" + std::to_string(index);
    //        qDebug() << QString::fromStdString(lineNo) << line.length();
    //        pcl::PointXYZI start, end;
    //        start.getVector3fMap() = line.start();
    //        end.getVector3fMap() = line.end();
    //        Eigen::Vector3f dir = line.direction();
    //        //m_cloudViewer->visualizer()->addArrow(end, start, r, g, b, 0, lineNo);
    //        m_cloudViewer->visualizer()->addLine(start, end, r, g, b, lineNo);
    //        m_cloudViewer->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 4, lineNo);
    //    }
    //    //break;
    //}

    //for (QList<LineSegment>::iterator i = clusterLines.begin(); i != clusterLines.end(); i++)
    //{
    //    double r = rand() * 1.0 / RAND_MAX;
    //    double g = rand() * 1.0 / RAND_MAX;
    //    double b = rand() * 1.0 / RAND_MAX;

    //    LineSegment line = *i;
    //    std::string lineNo = "line_" + std::to_string(clusterIndex);
    //    pcl::PointXYZI start, end;
    //    start.getVector3fMap() = line.start();
    //    end.getVector3fMap() = line.end();
    //    Eigen::Vector3f dir = line.direction();
    //    //m_cloudViewer->visualizer()->addArrow(end, start, r, g, b, 0, lineNo);
    //    m_cloudViewer->visualizer()->addLine(start, end, r, g, b, lineNo);
    //    m_cloudViewer->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 4, lineNo);
    //    clusterIndex++;
    //    //break;
    //}

    //for (std::vector<pcl::PointIndices>::const_iterator it = indices.begin(); it != indices.end(); ++it, clusterIndex++)
    //{
    //    qDebug() << it->indices.size();

    //    double r = rand() * 1.0 / RAND_MAX;
    //    double g = rand() * 1.0 / RAND_MAX;
    //    double b = rand() * 1.0 / RAND_MAX;

    //    for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
    //    {
    //        int index = *pit;
    //        std::string lineNo = "line_" + std::to_string(index);
    //        clusterCloud->points[index].intensity = clusterIndex;

    //        qDebug() << "    " << QString::fromStdString(lineNo);

    //        pcl::PointXYZI start, end;
    //        start.getVector3fMap() = lines[index].start();
    //        end.getVector3fMap() = lines[index].end();
    //        Eigen::Vector3f dir = lines[index].direction();
    //        //m_cloudViewer->visualizer()->addArrow(end, start, r, g, b, 0, lineNo);
    //        m_cloudViewer->visualizer()->addLine(start, end, r, g, b, lineNo);
    //        m_cloudViewer->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, lineNo);

    //    }
    //}
    //Eigen::Affine3f t;
    //pcl::getTransformation(-2.0, 0.0, 0.0, 0.0, 0.0, 0.0, t);
    //pcl::transformPointCloud(*clusterCloud, *clusterCloud, t);
    //pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> iColor(clusterCloud, "intensity");
    ////pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZI> redColorCluster(clusterCloud, 255, 0, 0);
    //m_cloudViewer->visualizer()->addPointCloud(clusterCloud, iColor, "cluster points");
    //m_cloudViewer->visualizer()->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "cluster points");

    pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZI> iColorB(m_boundaryCloud, "intensity");
    m_cloudViewer->visualizer()->addPointCloud(m_boundaryCloud, iColorB, "boundary points");

//    for (size_t i = 0; i < lines.size() && false; i++)
//    {
//        std::string lineNo = "line_" + std::to_string(i);
//
//        double r = rand() * 1.0 / RAND_MAX;
//        double g = rand() * 1.0 / RAND_MAX;
//        double b = rand() * 1.0 / RAND_MAX;
//        //double r = 1;
//        //double g = 0;
//        //double b = 0;
//
////        qDebug().noquote().nospace() << QString::fromStdString(lineNo) << " length is " << lines[i].length() << "m";
//
//        if (lines[i].length() > 0.f)
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
//
//    QList<LineCluster*> clusters = le.getLineClusters();
//    for (int i = 0; i < clusters.size() && false; i++)
//    {
//        double r = rand() * 1.0 / RAND_MAX;
//        double g = rand() * 1.0 / RAND_MAX;
//        double b = rand() * 1.0 / RAND_MAX;
//
//        LineSegment line = clusters[i]->merge();
//        if (line.length() < 0.1f)
//            continue;
//        std::string lineNo = "cluster_line_" + std::to_string(i);
//        pcl::PointXYZI start, end;
//        start.getVector3fMap() = line.start();
//        end.getVector3fMap() = line.end();
//        m_cloudViewer->visualizer()->addLine(start, end, r, g, b, lineNo);
//        m_cloudViewer->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 2, lineNo);
//
//        for (int j = 0; clusters[i]->size() > 1 && j < clusters[i]->size(); j++)
//        {
//            std::string lineNo = "cluster_line_" + std::to_string(i) + "_" + std::to_string(j);
//            pcl::PointXYZI start, end;
//            start.getVector3fMap() = clusters[i]->lines()[j].start();
//            end.getVector3fMap() = clusters[i]->lines()[j].end();
//    //        m_cloudViewer->visualizer()->addArrow(end, start, r, g, b, 0, lineNo);
//            m_cloudViewer->visualizer()->addLine(start, end, r, g, b, lineNo);
//            m_cloudViewer->visualizer()->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 4, lineNo);
//        }
//    }
    StopWatch::instance().tock("show_result");

}

void LineMatchCudaOdometry::afterProcessing(Frame& frame)
{
}
