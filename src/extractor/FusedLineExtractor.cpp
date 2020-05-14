#include "FusedLineExtractor.h"

#include <opencv2/cudafilters.hpp>
#include <pcl/common/pca.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>

#include "util/Utils.h"
#include "util/StopWatch.h"
#include "cuda/cuda.hpp"
#include "cuda/FusedLineInternal.h"
#include "EDLines.h"
#include "device/SensorReaderDevice.h"
#include "common/Parameters.h"

FusedLineExtractor::FusedLineExtractor(QObject* parent)
    : QObject(parent),
    m_init(false)
{

}

FusedLineExtractor::~FusedLineExtractor()
{
}

void FusedLineExtractor::computeGPU(cuda::FusedLineFrame& frame)
{
    cv::Ptr<cv::cuda::Filter> filter;

    cv::cuda::cvtColor(frame.rgbMatGpu, frame.grayMatGpu, cv::COLOR_RGB2GRAY);

    cv::cuda::GpuMat gaussianMat;
    filter = cv::cuda::createGaussianFilter(frame.grayMatGpu.type(), frame.grayMatGpu.type(), cv::Size(), 1.5);
    filter->apply(frame.grayMatGpu, frame.grayMatGpu);
    //gaussianMat.copyTo(frame.grayMatGpu);

    cuda::extractEDlines(frame);
    
    cv::Mat anchorImage;
    frame.anchorMatGpu.download(anchorImage);

    //cv::Mat angleImage;
    //frame.angleMatGpu.download(angleImage);
    
    //cv::Mat radiusImage;
    //frame.radiusMatGpu.download(radiusImage);

    //std::vector<float3> projVects;
    //frame.projCloud.download(projVects);

    //pcl::PointCloud<pcl::PointXYZI>::Ptr projCloud(new pcl::PointCloud<pcl::PointXYZI>);
    ////std::vector<int> projIndices;

    //for (int j = 0; j < frame.parameters.rgbHeight; j++)
    //{
    //    for (int i = 0; i < frame.parameters.rgbWidth; i++)
    //    {
    //        int index = j * frame.parameters.rgbWidth + i;
    //        uchar anchor = anchorImage.ptr<uchar>(j)[i];
    //        if (anchor == ANCHOR_PIXEL)
    //        {
    //            pcl::PointXYZI point;
    //            float3 vec = projVects[index];
    //            point.x = vec.x;
    //            point.y = vec.y;
    //            point.z = vec.z;
    //            point.intensity = index;
    //            projCloud->points.push_back(point);
    //        }
    //    }
    //}

    //pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
    //tree->setInputCloud(projCloud);

    //pcl::IndicesClusters clusters;
    //pcl::EuclideanClusterExtraction<pcl::PointXYZI> ece;
    //ece.setClusterTolerance(0.01f);
    //ece.setInputCloud(projCloud);
    //ece.setMinClusterSize(30);
    //ece.setMaxClusterSize(projCloud->points.size());
    //ece.setSearchMethod(tree);
    //ece.extract(clusters);

    //cv::Mat showImg(frame.parameters.rgbHeight, frame.parameters.rgbWidth, CV_8UC3, cv::Scalar(0));
    //for (int i = 0; i < clusters.size(); i++)
    //{
    //    std::cout << "cluster " << i << ": " << clusters[i].indices.size() << std::endl;
    //    for (int j = 0; j < clusters[i].indices.size(); j++)
    //    {
    //        int index = projCloud->points[clusters[i].indices[j]].intensity;
    //        int ix = index % frame.parameters.rgbWidth;
    //        int iy = index / frame.parameters.rgbWidth;
    //        showImg.ptr<uchar>(iy)[ix * 3] = i * 4;
    //        showImg.ptr<uchar>(iy)[ix * 3 + 1] = 255;
    //        showImg.ptr<uchar>(iy)[ix * 3 + 2] = 255;
    //        //anchorImage.ptr<uchar>(iy)[ix] = i * 4;
    //    }
    //}
    //cv::cvtColor(showImg, showImg, cv::COLOR_HSV2RGB);
    //cv::imshow("showImage", showImg);
}

void FusedLineExtractor::compute(Frame& frame, cuda::GpuFrame& frameGpu)
{
    // 抽取edline直线
    TICK("edlines");
    cv::Mat grayImage;
    cv::cvtColor(frame.colorMat(), grayImage, cv::COLOR_RGB2GRAY);
    EDLines lineHandler = EDLines(grayImage, SOBEL_OPERATOR);
    TOCK("edlines");
    cv::Mat linesMat = lineHandler.getLineImage();
    cv::Mat edlinesMat = lineHandler.drawOnImage();
    cv::imshow("ed lines", edlinesMat);
    cv::imshow("lines", linesMat);
    int linesCount = lineHandler.getLinesNo();
    std::vector<LS> lines = lineHandler.getLines();

    // 抽取出的直线集合放在这儿
     m_groupPoints.clear();
    for (int i = 0; i < linesCount; i++)
    {
        m_groupPoints.insert(i, pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>));
    }

    // 用cuda抽取be点和折线点
    frameGpu.parameters.cx = frame.getDevice()->cx();
    frameGpu.parameters.cy = frame.getDevice()->cy();
    frameGpu.parameters.fx = frame.getDevice()->fx();
    frameGpu.parameters.fy = frame.getDevice()->fy();
    frameGpu.parameters.minDepth = Settings::BoundaryExtractor_MinDepth.value();
    frameGpu.parameters.maxDepth = Settings::BoundaryExtractor_MaxDepth.value();
    frameGpu.parameters.borderLeft = Settings::BoundaryExtractor_BorderLeft.intValue();
    frameGpu.parameters.borderRight = Settings::BoundaryExtractor_BorderRight.intValue();
    frameGpu.parameters.borderTop = Settings::BoundaryExtractor_BorderTop.intValue();
    frameGpu.parameters.borderBottom = Settings::BoundaryExtractor_BorderBottom.intValue();
    frameGpu.parameters.depthShift = 1000;
    frameGpu.parameters.normalKernalRadius = Settings::BoundaryExtractor_CudaNormalKernalRadius.intValue();
    frameGpu.parameters.normalKnnRadius = Settings::BoundaryExtractor_CudaNormalKnnRadius.value();
    frameGpu.parameters.boundaryEstimationRadius = Settings::BoundaryExtractor_CudaBEKernalRadius.intValue();
    frameGpu.parameters.boundaryGaussianSigma = Settings::BoundaryExtractor_CudaGaussianSigma.value();
    frameGpu.parameters.boundaryGaussianRadius = Settings::BoundaryExtractor_CudaGaussianKernalRadius.intValue();
    frameGpu.parameters.boundaryEstimationDistance = Settings::BoundaryExtractor_CudaBEDistance.value();
    frameGpu.parameters.boundaryAngleThreshold = Settings::BoundaryExtractor_CudaBEAngleThreshold.value();
    frameGpu.parameters.classifyRadius = Settings::BoundaryExtractor_CudaClassifyKernalRadius.intValue();
    frameGpu.parameters.classifyDistance = Settings::BoundaryExtractor_CudaClassifyDistance.value();
    frameGpu.parameters.peakClusterTolerance = Settings::BoundaryExtractor_CudaPeakClusterTolerance.intValue();
    frameGpu.parameters.minClusterPeaks = Settings::BoundaryExtractor_CudaMinClusterPeaks.intValue();
    frameGpu.parameters.maxClusterPeaks = Settings::BoundaryExtractor_CudaMaxClusterPeaks.intValue();
    frameGpu.parameters.cornerHistSigma = Settings::BoundaryExtractor_CudaCornerHistSigma.value();

    cv::cuda::GpuMat boundaryMatGpu(frame.getDepthHeight(), frame.getDepthWidth(), CV_8U, frameGpu.boundaryImage);
    cv::cuda::GpuMat pointsMatGpu(frame.getDepthHeight(), frame.getDepthWidth(), CV_32S, frameGpu.indicesImage);

    TICK("extracting_boundaries");
    cuda::generatePointCloud(frameGpu);
    TOCK("extracting_boundaries");

    TICK("boundaries_downloading");
    boundaryMatGpu.download(m_boundaryMat);
    pointsMatGpu.download(m_pointsMat);
    std::vector<float3> points;
    frameGpu.pointCloud.download(points);
    std::vector<float3> normals;
    frameGpu.pointCloudNormals.download(normals);
    std::vector<uchar> boundaries;
    frameGpu.boundaries.download(boundaries);

    // 开始2d和3d的比对。
    m_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    m_allBoundary.reset(new pcl::PointCloud<pcl::PointXYZI>);
    m_normals.reset(new pcl::PointCloud<pcl::Normal>);
    int negativeNum = 0;
    for(int i = 0; i < frame.getDepthHeight(); i++) 
    {
        for(int j = 0; j < frame.getDepthWidth(); j++) 
        {
            cv::Point coord(j, i);
            int index = i * frame.getDepthWidth() + j;
            float3 value = points[index];
            uchar pointType = boundaries[index];
            ushort lineNo = linesMat.ptr<ushort>(i)[j];
            pcl::PointXYZ pt;
            pcl::PointXYZI ptI;
            pcl::Normal normal;

            pt.x = value.x;
            pt.y = value.y;
            pt.z = value.z;
            ptI.x = value.x;
            ptI.y = value.y;
            ptI.z = value.z;
            ptI.intensity = lineNo;

            normal.normal_x = normals[index].x;
            normal.normal_y = normals[index].y;
            normal.normal_z = normals[index].z;

            int ptIndex = m_pointsMat.at<int>(coord);
            if (ptIndex < 0)
            {
                negativeNum++;
            }
            else
            {
                m_cloud->push_back(pt);
                m_normals->push_back(normal);
                ptIndex -= negativeNum;
                m_pointsMat.at<int>(coord) = ptIndex;
            }
            
            //std::cout << j << ", " << i << ": " << lineNo << std::endl;

            if (pointType > 0 && lineNo != 65535)
            {
                //Eigen::Vector2f pt2d(j, i);
                //LS line = lines[lineNo];
                //cv::Point cvLineDir = line.end - line.start;
                //Eigen::Vector2f lineDir(cvLineDir.x, cvLineDir.y);
                //lineDir.normalize();
                //Eigen::Vector2f vLineDir(lineDir.x(), -lineDir.y());
                //vLineDir.normalize();
                //Eigen::Vector3f avg(Eigen::Vector3f::Zero());
                //int count = 0;
                //for (int ni = -2; ni <= 2; ni++)
                //{
                //    for (int nj = -2; nj <= 2; nj++)
                //    {
                //        Eigen::Vector2f pt2dN = pt2d + vLineDir * ni + lineDir * nj;
                //        Eigen::Vector2i pt2dNI = pt2dN.cast<int>();
                //        if (pt2dNI.x() < 0 || pt2dNI.x() >= frame.getDepthWidth() || pt2dNI.y() < 0 || pt2dNI.y() >= frame.getDepthHeight())
                //            continue;
                //        int ptIndexN = pt2dNI.y() * frame.getDepthHeight() + pt2dNI.x();
                //        float3 valueN = points[ptIndexN];
                //        uchar pointTypeN = boundaries[ptIndexN];
                //        if (pointTypeN <= 0)
                //            continue;
                //        avg += toVector3f(valueN);
                //        count++;
                //    }
                //}
                //avg /= count;
                ////std::cout << j << ", " << i << ": " << lineNo << ", count = " << count << ", avg = " << avg.transpose() << std::endl;

                //if (ptI.z <= avg.z())
                //{
                    m_allBoundary->points.push_back(ptI);
                    m_groupPoints[lineNo]->points.push_back(ptI);
                //}
            }
        }
    }
    m_cloud->width = m_cloud->points.size();
    m_cloud->height = 1;
    m_cloud->is_dense = true;
    m_normals->width = m_normals->points.size();
    m_normals->height = 1;
    m_normals->is_dense = true;
    m_allBoundary->width = m_allBoundary->points.size();
    m_allBoundary->height = 1;
    m_allBoundary->is_dense = true;

    m_lines.clear();

    for (int i = 0; i < linesCount; i++)
    {
        if (m_groupPoints[i]->size() < 10)
            continue;

        Eigen::Vector3f gCenter(Eigen::Vector3f::Zero());
        for (int j = 0; j < m_groupPoints[i]->points.size(); j++)
        {
            Eigen::Vector3f np = m_groupPoints[i]->points[j].getArray3fMap();
            gCenter += np;
        }
        gCenter /= m_groupPoints[i]->points.size();

        // 因为同一直线编号的点集中可能既有真点也有veil点，所以先做区域分割。
        pcl::IndicesClusters clusters;
        pcl::EuclideanClusterExtraction<pcl::PointXYZI> ece;
        ece.setClusterTolerance(0.05f);
        ece.setInputCloud(m_groupPoints[i]);
        ece.setMinClusterSize(1);
        ece.setMaxClusterSize(m_groupPoints[i]->points.size());
        ece.extract(clusters);

        std::cout << i << ": " << "count = " << m_groupPoints[i]->points.size() << std::endl;

        int maxSize = 0;
        int maxIndex = 0;
        // 分割后，找出点数最多的子区域作为初始内点集合。即cloud。
        for (int j = 0; j < clusters.size(); j++)
        {
            Eigen::Vector3f clusterCenter(Eigen::Vector3f::Zero());
            for (int n = 0; n < clusters[j].indices.size(); n++)
            {
                Eigen::Vector3f np = m_groupPoints[i]->points[clusters[j].indices[n]].getArray3fMap();
                clusterCenter += np;
            }
            clusterCenter /= clusters[j].indices.size();
            bool valid = true;
            if (clusterCenter.z() > gCenter.z())
            {
                float dist = clusterCenter.z() - gCenter.z();
                if (dist > 0.03f)
                {
                    valid = false;
                }
            }
            std::cout << "  sub " << j << ", count = " << clusters[j].indices.size() << ", farer = " << (clusterCenter.z() > gCenter.z()) << ", z dist = " << (clusterCenter.z() - gCenter.z())
                << ", valid = " << valid << std::endl;
            if (valid && clusters[j].indices.size() > maxSize)
            {
                maxSize = clusters[j].indices.size();
                maxIndex = j;
            }
        }
        if (maxSize < 3)
            continue;

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::copyPointCloud(*m_groupPoints[i], clusters[maxIndex].indices, *cloud);
        //pcl::copyPointCloud(*m_groupPoints[i], *cloud);

        // 计算这个初始内点集合的主方向和中点。
        pcl::PCA<pcl::PointXYZI> pca;
        pca.setInputCloud(cloud);
        Eigen::Vector3f eigenValues = pca.getEigenValues();
        float sqrt1 = sqrt(eigenValues[0]);
        float sqrt2 = sqrt(eigenValues[1]);
        float sqrt3 = sqrt(eigenValues[2]);
        float a1 = (sqrt1 - sqrt2) / sqrt1;
        float a2 = (sqrt2 - sqrt3) / sqrt1;
        float a3 = sqrt3 / sqrt1;

        std::cout << "  " << m_groupPoints[i]->size() << ", cluster: " << clusters.size() << ", a1 = " << a1 << ", a2 = " << a2 << ", a3 = " << a3 << std::endl;
        std::cout << "  init inliers size: " << cloud->size() << std::endl;

        // 主方向
        Eigen::Vector3f dir = pca.getEigenVectors().col(0).normalized();
        // 中点
        Eigen::Vector3f center = pca.getMean().head(3);

        // 然后，遍历剩余的子区域点集，查看每一个点到这条直线的距离是否在阈值以内，在就加到内点集中，不在就抛弃。
        for (int j = 0; j < clusters.size(); j++)
        {
            if (j == maxIndex)
                continue;

            for (int n = 0; n < clusters[j].indices.size(); n++)
            {
                int nIndex = clusters[j].indices[n];
                pcl::PointXYZI pclPt = m_groupPoints[i]->points[nIndex];
                Eigen::Vector3f pt = pclPt.getArray3fMap();
                float dist = (pt - center).cross(dir).norm();
                // 暂时硬编码的阈值。
                if (dist <= 0.05f)
                {
                    cloud->points.push_back(pclPt);
                }
            }
        }
        if (cloud->size() < 10)
            continue;

        // 最后再计算一遍内点集的主方向与中点。
        std::cout << "    final: " << cloud->size() << ", max size: " << maxSize << ", max index: " << maxIndex << std::endl;
        //std::cout << "    final: " << cloud->size() << std::endl;
        pcl::PCA<pcl::PointXYZI> pcaFinal;
        pcaFinal.setInputCloud(cloud);
        eigenValues = pcaFinal.getEigenValues();
        dir = pcaFinal.getEigenVectors().col(0).normalized();
        center = pcaFinal.getMean().head(3);
        //Eigen::Vector3f eigenValues = pcaFinal.getEigenValues();
        //Eigen::Vector3f dir = pcaFinal.getEigenVectors().col(0).normalized();
        //Eigen::Vector3f center = pcaFinal.getMean().head(3);

        // 确定端点。
        Eigen::Vector3f start(0, 0, 0);
        Eigen::Vector3f end(0, 0, 0);
        for (int j = 0; j < cloud->size(); j++)
        {
            pcl::PointXYZI& ptBoundary = cloud->points[j];
            Eigen::Vector3f boundaryPoint = ptBoundary.getVector3fMap();
            Eigen::Vector3f projPoint = closedPointOnLine(boundaryPoint, dir, center);

            if (start.isZero())
            {
                // 如果第一次循环，让当前点作为起点
                start = projPoint;
            }
            else
            {
                // 将当前点与当前计算出的临时起点连在一起，查看其与当前聚集主方向的一致性，若一致则当前点为新的起点。
                if ((start - projPoint).dot(dir) > 0)
                {
                    start = projPoint;
                }
            }

            if (end.isZero())
            {
                // 如果第一次循环，让当前点作为终点
                end = projPoint;
            }
            else
            {
                // 将当前点与当前计算出的临时终点连在一起，查看其与当前聚集主方向的一致性，若一致则当前点为新的终点。
                if ((projPoint - end).dot(dir) > 0)
                {
                    end = projPoint;
                }
            }
        }
        //start = closedPointOnLine(start, dir, center);
        //end = closedPointOnLine(end, dir, center);

        LS3D line;
        line.start.getArray3fMap() = start;
        line.end.getArray3fMap() = end;
        line.center.getArray3fMap() = center;
        if (line.length() > 0.1f)
            m_lines.insert(i, line);
    }
    
    qDebug() << "all boundary points:" << m_allBoundary->size();
    TOCK("boundaries_downloading");
}
