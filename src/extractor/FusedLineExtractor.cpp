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
    // ��ȡedlineֱ��
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

    // ��ȡ����ֱ�߼��Ϸ������
     m_groupPoints.clear();
    for (int i = 0; i < linesCount; i++)
    {
        m_groupPoints.insert(i, pcl::PointCloud<pcl::PointXYZI>::Ptr(new pcl::PointCloud<pcl::PointXYZI>));
    }

    // ��cuda��ȡbe������ߵ�
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

    // ��ʼ2d��3d�ıȶԡ�
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
                m_allBoundary->points.push_back(ptI);
                m_groupPoints[lineNo]->points.push_back(ptI);
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

        // ��Ϊͬһֱ�߱�ŵĵ㼯�п��ܼ������Ҳ��veil�㣬������������ָ
        pcl::IndicesClusters clusters;
        pcl::EuclideanClusterExtraction<pcl::PointXYZI> ece;
        ece.setClusterTolerance(0.05f);
        ece.setInputCloud(m_groupPoints[i]);
        ece.setMinClusterSize(1);
        ece.setMaxClusterSize(m_groupPoints[i]->points.size());
        ece.extract(clusters);

        int maxSize = 0;
        int maxIndex = 0;
        // �ָ���ҳ�����������������Ϊ��ʼ�ڵ㼯�ϡ���cloud��
        for (int j = 0; j < clusters.size(); j++)
        {
            if (clusters.size() > maxSize)
            {
                maxSize = clusters.size();
                maxIndex = j;
            }
        }
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
        pcl::copyPointCloud(*m_groupPoints[i], clusters[maxIndex].indices, *cloud);

        // ���������ʼ�ڵ㼯�ϵ���������е㡣
        pcl::PCA<pcl::PointXYZI> pca;
        pca.setInputCloud(cloud);
        Eigen::Vector3f eigenValues = pca.getEigenValues();
        float sqrt1 = sqrt(eigenValues[0]);
        float sqrt2 = sqrt(eigenValues[1]);
        float sqrt3 = sqrt(eigenValues[2]);
        float a1 = (sqrt1 - sqrt2) / sqrt1;
        float a2 = (sqrt2 - sqrt3) / sqrt1;
        float a3 = sqrt3 / sqrt1;

        std::cout << i << ": " << "count = " << m_groupPoints[i]->size() << ", cluster: " << clusters.size() << ", a1 = " << a1 << ", a2 = " << a2 << ", a3 = " << a3 << std::endl;

        // ������
        Eigen::Vector3f dir = pca.getEigenVectors().col(0).normalized();
        // �е�
        Eigen::Vector3f center = pca.getMean().head(3);

        // Ȼ�󣬱���ʣ���������㼯���鿴ÿһ���㵽����ֱ�ߵľ����Ƿ�����ֵ���ڣ��ھͼӵ��ڵ㼯�У����ھ�������
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
                // ��ʱӲ�������ֵ��
                if (dist <= 0.05f)
                {
                    cloud->points.push_back(pclPt);
                }
            }
        }
        if (cloud->size() < 10)
            continue;

        // ����ټ���һ���ڵ㼯�����������е㡣
        std::cout << "    final: " << cloud->size() << std::endl;
        pcl::PCA<pcl::PointXYZI> pcaFinal;
        pca.setInputCloud(cloud);
        eigenValues = pca.getEigenValues();
        dir = pca.getEigenVectors().col(0).normalized();
        center = pca.getMean().head(3);

        // ȷ���˵㡣
        Eigen::Vector3f start(0, 0, 0);
        Eigen::Vector3f end(0, 0, 0);
        for (int j = 0; j < cloud->size(); j++)
        {
            pcl::PointXYZI& ptBoundary = cloud->points[j];
            Eigen::Vector3f boundaryPoint = ptBoundary.getVector3fMap();
            Eigen::Vector3f projPoint = closedPointOnLine(boundaryPoint, dir, center);

            if (start.isZero())
            {
                // �����һ��ѭ�����õ�ǰ����Ϊ���
                start = projPoint;
            }
            else
            {
                // ����ǰ���뵱ǰ���������ʱ�������һ�𣬲鿴���뵱ǰ�ۼ��������һ���ԣ���һ����ǰ��Ϊ�µ���㡣
                if ((start - projPoint).dot(dir) > 0)
                {
                    start = projPoint;
                }
            }

            if (end.isZero())
            {
                // �����һ��ѭ�����õ�ǰ����Ϊ�յ�
                end = projPoint;
            }
            else
            {
                // ����ǰ���뵱ǰ���������ʱ�յ�����һ�𣬲鿴���뵱ǰ�ۼ��������һ���ԣ���һ����ǰ��Ϊ�µ��յ㡣
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
