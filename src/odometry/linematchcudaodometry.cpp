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
        m_frameGpu.allocate();
        m_frameGpu.parameters.cx = frame.getDevice()->cx();
        m_frameGpu.parameters.cy = frame.getDevice()->cy();
        m_frameGpu.parameters.fx = frame.getDevice()->fx();
        m_frameGpu.parameters.fy = frame.getDevice()->fy();
        m_frameGpu.parameters.colorWidth = frame.getColorWidth();
        m_frameGpu.parameters.colorHeight = frame.getColorHeight();
        m_frameGpu.parameters.depthWidth = frame.getDepthWidth();
        m_frameGpu.parameters.depthHeight = frame.getDepthHeight();
        m_frameGpu.parameters.depthShift = frame.getDevice()->depthShift();

        m_cloud = pcl::PointCloud<pcl::PointXYZRGB>::Ptr(new pcl::PointCloud<pcl::PointXYZRGB>());
        m_normals = pcl::PointCloud<pcl::Normal>::Ptr(new pcl::PointCloud<pcl::Normal>());

        if (!m_boundaryExtractor)
        {
            m_boundaryExtractor.reset(new BoundaryExtractor);
        }

        if (!m_lineExtractor)
        {
            m_lineExtractor.reset(new LineExtractor);
        }

        if (!m_lineMatcher)
        {
            m_lineMatcher.reset(new LineMatcher);
        }

        m_lineMatcher->setMaxIterations(Settings::LineMatcher_MaxIterations.intValue());

        m_lineExtractor->setBoundaryCloudA1dThreshold(Settings::LineExtractor_BoundaryCloudA1dThreshold.value());
        m_lineExtractor->setCornerCloudA1dThreshold(Settings::LineExtractor_CornerCloudA1dThreshold.value());
        m_lineExtractor->setBoundaryCloudSearchRadius(Settings::LineExtractor_BoundaryCloudSearchRadius.value());
        m_lineExtractor->setCornerCloudSearchRadius(Settings::LineExtractor_CornerCloudSearchRadius.value());
        m_lineExtractor->setPCASearchRadius(Settings::LineExtractor_PCASearchRadius.value());
        m_lineExtractor->setMinNeighboursCount(Settings::LineExtractor_MinNeighboursCount.intValue());
        m_lineExtractor->setAngleCloudSearchRadius(Settings::LineExtractor_AngleCloudSearchRadius.value());
        m_lineExtractor->setAngleCloudMinNeighboursCount(Settings::LineExtractor_AngleCloudMinNeighboursCount.intValue());
        m_lineExtractor->setMinLineLength(Settings::LineExtractor_MinLineLength.value());
        m_lineExtractor->setBoundaryLineInterval(Settings::LineExtractor_BoundaryLineInterval.value());
        m_lineExtractor->setCornerLineInterval(Settings::LineExtractor_CornerLineInterval.value());
        m_lineExtractor->setBoundaryMaxZDistance(Settings::LineExtractor_BoundaryMaxZDistance.value());
        m_lineExtractor->setCornerMaxZDistance(Settings::LineExtractor_CornerMaxZDistance.value());
        m_lineExtractor->setCornerGroupLinesSearchRadius(Settings::LineExtractor_CornerGroupLinesSearchRadius.value());

        m_boundaryExtractor.reset(new BoundaryExtractor);
        m_boundaryExtractor->setBorderLeft(Settings::BoundaryExtractor_BorderLeft.value());
        m_boundaryExtractor->setBorderRight(Settings::BoundaryExtractor_BorderRight.value());
        m_boundaryExtractor->setBorderTop(Settings::BoundaryExtractor_BorderTop.value());
        m_boundaryExtractor->setBorderBottom(Settings::BoundaryExtractor_BorderBottom.value());
        m_boundaryExtractor->setMinDepth(Settings::BoundaryExtractor_MinDepth.value());
        m_boundaryExtractor->setMaxDepth(Settings::BoundaryExtractor_MaxDepth.value());
        m_boundaryExtractor->setCudaNormalKernalRadius(Settings::BoundaryExtractor_CudaNormalKernalRadius.intValue());
        m_boundaryExtractor->setCudaNormalKnnRadius(Settings::BoundaryExtractor_CudaNormalKnnRadius.value());
        m_boundaryExtractor->setCudaBEDistance(Settings::BoundaryExtractor_CudaBEDistance.value());
        m_boundaryExtractor->setCudaBEAngleThreshold(Settings::BoundaryExtractor_CudaBEAngleThreshold.value());
        m_boundaryExtractor->setCudaBEKernalRadius(Settings::BoundaryExtractor_CudaBEKernalRadius.value());
        m_boundaryExtractor->setCudaGaussianSigma(Settings::BoundaryExtractor_CudaGaussianSigma.value());
        m_boundaryExtractor->setCudaGaussianKernalRadius(Settings::BoundaryExtractor_CudaGaussianKernalRadius.value());
        m_boundaryExtractor->setCudaClassifyKernalRadius(Settings::BoundaryExtractor_CudaClassifyKernalRadius.value());
        m_boundaryExtractor->setCudaClassifyDistance(Settings::BoundaryExtractor_CudaClassifyDistance.value());
        m_boundaryExtractor->setCudaPeakClusterTolerance(Settings::BoundaryExtractor_CudaPeakClusterTolerance.intValue());
        m_boundaryExtractor->setCudaMinClusterPeaks(Settings::BoundaryExtractor_CudaMinClusterPeaks.intValue());
        m_boundaryExtractor->setCudaMaxClusterPeaks(Settings::BoundaryExtractor_CudaMaxClusterPeaks.intValue());
        m_boundaryExtractor->setCudaCornerHistSigma(Settings::BoundaryExtractor_CudaCornerHistSigma.value());

        m_pose = Eigen::Matrix4f::Identity();

        m_init = true;
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

    pcl::IndicesPtr indices(new std::vector<int>);

    m_cloud = frame.getCloud(*indices);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::copyPointCloud<pcl::PointXYZRGB, pcl::PointXYZ>(*m_cloud, *cloud);

    cv::cuda::GpuMat colorMatGpu(frame.getColorHeight(), frame.getColorWidth(), CV_8UC3, m_frameGpu.colorImage);
    cv::cuda::GpuMat depthMatGpu(frame.getDepthHeight(), frame.getDepthWidth(), CV_16U, m_frameGpu.depthImage);
    colorMatGpu.upload(frame.colorMat());
    depthMatGpu.upload(frame.depthMat());

    pcl::PointCloud<pcl::PointXYZI>::Ptr boundaryPoints;
    cv::Mat pointsMat;
    pcl::PointCloud<pcl::Normal>::Ptr normals;
    pcl::PointCloud<pcl::PointXYZI>::Ptr cornerPoints;
    m_boundaryExtractor->setInputCloud(cloud);
    m_boundaryExtractor->setWidth(frame.getDepthWidth());
    m_boundaryExtractor->setHeight(frame.getDepthHeight());
    m_boundaryExtractor->setCx(frame.getDevice()->cx());
    m_boundaryExtractor->setCy(frame.getDevice()->cy());
    m_boundaryExtractor->setFx(frame.getDevice()->fx());
    m_boundaryExtractor->setFy(frame.getDevice()->fy());
    m_boundaryExtractor->setNormals(nullptr);
    m_boundaryExtractor->computeCUDA(m_frameGpu);
    boundaryPoints = m_boundaryExtractor->boundaryPoints();
    pointsMat = m_boundaryExtractor->pointsMat();
    normals = m_boundaryExtractor->normals();
    cornerPoints = m_boundaryExtractor->cornerPoints();

    QList<LineSegment> lineSegments;
    lineSegments = m_lineExtractor->compute(boundaryPoints, cornerPoints);
    pcl::PointCloud<Line>::Ptr lines = m_lineExtractor->lineCloud();

    float rotationError = 0;
    float transError = 0;
    if (m_frames.size() > 0)
    {
        pcl::PointCloud<Line>::Ptr lastLines = m_lines[m_frames.size() - 1];
        m_frames.append(frame);
        m_lines.append(lines);

        Eigen::Matrix4f M = m_lineMatcher->compute(lastLines, lines, rotationError, transError);
        m_poses.append(M);
        m_rotationErrors.append(rotationError);
        m_transErrors.append(transError);

        m_pose = M * m_pose;

        std::cout << m_pose << std::endl;
    }
}

void LineMatchCudaOdometry::afterProcessing(Frame& frame)
{
}
