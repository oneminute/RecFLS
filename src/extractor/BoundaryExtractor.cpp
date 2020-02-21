#include "BoundaryExtractor.h"

#include <QDebug>
#include <QtMath>

#include <pcl/common/common.h>
#include <pcl/features/boundary.h>
#include <pcl/features/impl/normal_3d.hpp>
#include <pcl/features/normal_3d.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/convolution_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/impl/extract_clusters.hpp>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>

#include "util/StopWatch.h"

BoundaryExtractor::BoundaryExtractor(QObject* parent)
    : QObject(parent)
    , m_downsamplingMethod(DM_VOXEL_GRID)
    , m_enableRemovalFilter(false)
    , m_downsampleLeafSize(0.005f)
    , m_outlierRemovalMeanK(50)
    , m_stddevMulThresh(1.0f)
    , m_gaussianSigma(4)
    , m_gaussianRSigma(4)
    , m_gaussianRadiusSearch(0.05f)
    , m_normalsRadiusSearch(0.05f)
    , m_boundaryRadiusSearch(0.1f)
    , m_boundaryAngleThreshold(M_PI_4)
    , m_matWidth(640)
    , m_matHeight(480)
    , m_depthShift(1000.f)
    , m_cx(320)
    , m_cy(240)
    , m_fx(583)
    , m_fy(583)
    , m_borderLeft(26)
    , m_borderRight(22)
    , m_borderTop(16)
    , m_borderBottom(16)
    , m_projectedRadiusSearch(M_PI / 72)
    , m_veilDistanceThreshold(0.075f)
    , m_crossPointsRadiusSearch(0.05f)
    , m_crossPointsClusterTolerance(0.1f)
    , m_curvatureThreshold(0.025f)
    , m_minNormalClusters(2)
    , m_maxNormalClusters(2)
    , m_planeDistanceThreshold(0.01)
    , m_classifyRadius(20)
    , m_planePointsRate(0.05f)
{

}

pcl::PointCloud<pcl::PointXYZI>::Ptr BoundaryExtractor::compute()
{
    // 重置点云数据指针
    m_allBoundary.reset(new pcl::PointCloud<pcl::PointXYZI>);
    m_downsampledCloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    m_filteredCloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    m_removalCloud.reset(new pcl::PointCloud<pcl::PointXYZ>);

    qDebug() << "downsampleLeafSize:" << m_downsampleLeafSize;

    if (!m_cloud || m_cloud->empty())
        return m_allBoundary;

    /*if (!m_indices)
    {
        m_indices.reset(new pcl::Indices);
    }

    if (m_indices->empty())
    {
        for (int i = 0; i < m_cloud->size(); i++)
        {
            m_indices->push_back(i);
        }
    }*/

    // 计算法线
    if (!m_normals || m_normals->empty())
    {
        computeNormals(m_cloud);
    }

    // 提取全部边界点
    boundaryEstimation(m_cloud);

    // 对所有的点进行分类
    classifyBoundaryPoints2();

    // 下采样
    m_downsampledCloud = downSampling(m_cloud);

    // 去除离群点
    m_removalCloud = outlierRemoval(m_downsampledCloud);

    // 高斯滤波
    m_filteredCloud = gaussianFilter(m_removalCloud);

    // 抽取平面
    extractPlanes();

    return m_allBoundary;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr BoundaryExtractor::computeCUDA(cuda::GpuFrame& frame)
{
    frame.parameters.cx = m_cx;
    frame.parameters.cy = m_cy;
    frame.parameters.fx = m_fx;
    frame.parameters.fy = m_fy;
    frame.parameters.minDepth = 0.4f;
    frame.parameters.maxDepth = 8.0f;
    frame.parameters.borderLeft = m_borderLeft;
    frame.parameters.borderRight = m_borderRight;
    frame.parameters.borderTop = m_borderTop;
    frame.parameters.borderBottom = m_borderBottom;
    frame.parameters.depthShift = m_depthShift;
    frame.parameters.normalKernelHalfSize = 20;
    frame.parameters.normalKernelMaxDistance = m_normalsRadiusSearch;
    frame.parameters.boundaryEstimationRadius = 20;
    frame.parameters.boundaryGaussianSigma = 4.f;
    frame.parameters.boundaryGaussianRadius = 20;
    frame.parameters.boundaryEstimationDistance = m_boundaryRadiusSearch;
    frame.parameters.boundaryAngleThreshold = m_boundaryAngleThreshold * 180 / M_PI;
    frame.parameters.classifyRadius = 20;
    frame.parameters.classifyDistance = 0.05f;

    cv::cuda::GpuMat indicesMatGpu(m_matHeight, m_matWidth, CV_32S, frame.indicesImage);
    cv::cuda::GpuMat boundaryMatGpu(m_matHeight, m_matWidth, CV_8U, frame.boundaryImage);
    cv::cuda::GpuMat boundaryIndicesMatGpu(m_matHeight, m_matWidth, CV_32S, frame.boundaryIndices);
    indicesMatGpu.setTo(cv::Scalar(-1));
    boundaryMatGpu.setTo(cv::Scalar(0));
    boundaryIndicesMatGpu.setTo(cv::Scalar(-1));

    TICK("extracting_boundaries");
    cuda::generatePointCloud(frame);
    TOCK("extracting_boundaries");

    TICK("boundaries_downloading");
    boundaryMatGpu.download(m_boundaryMat);
    std::vector<float3> points;
    frame.pointCloud.download(points);
    std::vector<float3> normals;
    frame.pointCloudNormals.download(normals);
    std::vector<uchar> boundaries;
    frame.boundaries.download(boundaries);

    m_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    m_allBoundary.reset(new pcl::PointCloud<pcl::PointXYZI>);
    m_boundaryPoints.reset(new pcl::PointCloud<pcl::PointXYZI>);
    m_veilPoints.reset(new pcl::PointCloud<pcl::PointXYZI>);
    m_borderPoints.reset(new pcl::PointCloud<pcl::PointXYZI>);
    m_normals.reset(new pcl::PointCloud<pcl::Normal>);
    for(int i = 0; i < m_matHeight; i++) 
    {
        for(int j = 0; j < m_matWidth; j++) 
        {
            int index = i * m_matWidth + j;
            float3 value = points[index];
            uchar pointType = boundaries[index];
            pcl::PointXYZ pt;
            pcl::PointXYZI ptI;
            pcl::Normal normal;

            pt.x = value.x;
            pt.y = value.y;
            pt.z = value.z;
            ptI.x = value.x;
            ptI.y = value.y;
            ptI.z = value.z;
            ptI.intensity = 1;

            normal.normal_x = normals[index].x;
            normal.normal_y = normals[index].y;
            normal.normal_z = normals[index].z;

            if (index % 1024 == 0)
            {
                //qDebug() << value.x << value.y << value.z;
            }

            if (pt.z > 0.4f && pt.z <= 8.0f) {
                m_cloud->push_back(pt);
                m_normals->push_back(normal);
            }

            if (pointType > 0)
            {
                m_allBoundary->points.push_back(ptI);
                if (pointType == 1)
                {
                    m_borderPoints->points.push_back(ptI);
                }
                else if (pointType == 2)
                {
                    m_veilPoints->points.push_back(ptI);
                }
                else if (pointType == 3)
                {
                    m_boundaryPoints->points.push_back(ptI);
                }
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
    m_boundaryPoints->width = m_boundaryPoints->points.size();
    m_boundaryPoints->height = 1;
    m_boundaryPoints->is_dense = true;
    m_veilPoints->width = m_veilPoints->points.size();
    m_veilPoints->height = 1;
    m_veilPoints->is_dense = true;
    m_borderPoints->width = m_borderPoints->points.size();
    m_borderPoints->height = 1;
    m_borderPoints->is_dense = true;
    TOCK("boundaries_downloading");

    // 下采样
    m_downsampledCloud = downSampling(m_cloud);

    // 去除离群点
    m_removalCloud = outlierRemoval(m_downsampledCloud);

    // 高斯滤波
    m_filteredCloud = gaussianFilter(m_removalCloud);

    // 抽取平面
    extractPlanes();
    
    return m_allBoundary;
}

void BoundaryExtractor::boundaryEstimation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    TICK("Boundary_Estimation");

    // 使用pcl的BE类提取边界点
    pcl::PointCloud<pcl::Boundary> boundary;
    pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> be;
    be.setInputCloud(cloud);
    //be.setIndices(m_indices);
    be.setInputNormals(m_normals);
    be.setRadiusSearch(m_boundaryRadiusSearch);
    be.setAngleThreshold(m_boundaryAngleThreshold);
    //be.setSearchMethod(tree);
    be.compute(boundary);

    // 将提取的点保存到m_allBoundary中
    for (int i = 0; i < boundary.points.size(); i++)
    {
        if (boundary[i].boundary_point == 1)
        {
            pcl::PointXYZI pt;
            pt.x = cloud->points[i].x;
            pt.y = cloud->points[i].y;
            pt.z = cloud->points[i].z;
            pt.intensity = 0;
            m_allBoundary->points.push_back(pt);
        }
    }
    m_allBoundary->width = m_allBoundary->points.size();
    m_allBoundary->height = 1;
    m_allBoundary->is_dense = true;
    qDebug() << "Extract boundary size:" << m_allBoundary->size() << ", from original cloud size:" << cloud->size();
    TOCK("Boundary_Estimation");
}

pcl::PointCloud<pcl::PointXYZ>::Ptr BoundaryExtractor::gaussianFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    TICK("Gaussian_Filter");
    // 高斯滤波
    pcl::PointCloud<pcl::PointXYZ>::Ptr out(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::filters::GaussianKernel<pcl::PointXYZ, pcl::PointXYZ>::Ptr kernel(new pcl::filters::GaussianKernel<pcl::PointXYZ, pcl::PointXYZ>);
    kernel->setSigma(m_gaussianSigma);
    kernel->setThresholdRelativeToSigma(m_gaussianRSigma);

    //Set up the Convolution Filter
    pcl::filters::Convolution3D<pcl::PointXYZ, pcl::PointXYZ, pcl::filters::GaussianKernel<pcl::PointXYZ, pcl::PointXYZ>> convolution;
    convolution.setKernel(*kernel);
    convolution.setInputCloud(cloud);
    //convolution.setIndices(m_indices);
    convolution.setRadiusSearch(m_gaussianRadiusSearch);
    convolution.convolve(*out);
    TOCK("Gaussian_Filter");
    return out;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr BoundaryExtractor::outlierRemoval(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    TICK("Outlier_Removal");
    // 剔除离群点
    std::vector<int> indices;
    pcl::PointCloud<pcl::PointXYZ>::Ptr out(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
    //sor.setIndices(m_indices);
    sor.setInputCloud(cloud);
    sor.setMeanK(m_outlierRemovalMeanK);
    sor.setStddevMulThresh(m_stddevMulThresh); //距离大于1倍标准方差
    sor.filter(indices);
    //m_indices.reset(new std::vector<int>(indices));
    pcl::copyPointCloud(*cloud, indices, *out);
    qDebug() << "Cloud size before outlier removal:" << cloud->size() << ", size after:" << out->size() << ", removed size:" << (cloud->size() - indices.size());
    
    TOCK("Outlier_Removal");
    return out;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr BoundaryExtractor::downSampling(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    TICK("Downsampling");
    pcl::PointCloud<pcl::PointXYZ>::Ptr out(new pcl::PointCloud<pcl::PointXYZ>);
    if (m_downsamplingMethod == DM_VOXEL_GRID)
    {
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        //vg.setIndices(m_indices);
        vg.setInputCloud(cloud);
        vg.setLeafSize(m_downsampleLeafSize, m_downsampleLeafSize, m_downsampleLeafSize);
        vg.filter(*out);
    }
    if (m_downsamplingMethod == DM_UNIFORM_SAMPLING)
    {
        pcl::UniformSampling<pcl::PointXYZ> us;
        //us.setIndices(m_indices);
        us.setInputCloud(cloud);
        us.setRadiusSearch(m_downsampleLeafSize);
        us.filter(*out);
    }

    qDebug() << "Size before downsampling:" << cloud->size() << ", size after:" << out->size();

    //m_indices->clear();
    /*for (int i = 0; i < m_downsampledCloud->size(); i++)
    {
    m_indices->push_back(i);
    }*/
    TOCK("Downsampling");
    return out;
}

void BoundaryExtractor::classifyBoundaryPoints()
{
    // 将边界点云反射映射为深度图，然后使用射线检测进行边界点分类。
    TICK("Classify_Boundary_Points");
    m_projectedCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    m_boundaryPoints.reset(new pcl::PointCloud<pcl::PointXYZI>);
    m_veilPoints.reset(new pcl::PointCloud<pcl::PointXYZI>);
    m_borderPoints.reset(new pcl::PointCloud<pcl::PointXYZI>);

    Eigen::Vector3f yAxis(0, 1, 0);
    Eigen::Vector3f zAxis(0, 0, 1);
    QList<float> lengthList;
    for (int index = 0; index < m_allBoundary->size(); index++)
    {
        pcl::PointXYZI point = m_allBoundary->points[index];
        Eigen::Vector3f ep = point.getVector3fMap();
        float length = ep.norm();
        ep.normalize();
        float projYLength = ep.dot(yAxis);
        float yRadians = qAcos(projYLength);
        //float zRadians = qAcos((ep - yAxis * projYLength).normalized().dot(zAxis));
        float zRadians = qAcos(ep.cross(yAxis).dot(zAxis));

        // 按照从深度图计算点云的公式反算回去
        int i = static_cast<int>(point.x * m_fx / point.z + m_cx);
        int j = static_cast<int>(point.y * m_fy / point.z + m_cy);
        
        // 直接判断当前点是否在图像边缘
        if (i <= m_borderLeft || j <= m_borderTop || i >= (m_matWidth - m_borderRight) || j >= (m_matHeight - m_borderBottom))
        {
            // 图像边框点
            m_borderPoints->push_back(point);
            continue;
        }
        else
        {
            pcl::PointXYZI projPoint;
            projPoint.x = zRadians;
            projPoint.y = yRadians;
            projPoint.z = 0;
            projPoint.intensity = index;
            m_projectedCloud->push_back(projPoint);
            lengthList.append(length);
        }
    }

    pcl::search::KdTree<pcl::PointXYZI> projTree;
    projTree.setInputCloud(m_projectedCloud);
    int isolates = 0;
    for (int i = 0; i < m_projectedCloud->size(); i++)
    {
        pcl::PointXYZI projPoint = m_projectedCloud->points[i];
        pcl::PointXYZI point = m_allBoundary->points[projPoint.intensity];
        float length = lengthList[i];

        std::vector<int> neighbourIndices;
        std::vector<float> neighbourDistances;
        projTree.radiusSearch(projPoint, m_projectedRadiusSearch, neighbourIndices, neighbourDistances);
        if (neighbourIndices.size() == 1)
        {
            // 孤点
            isolates++;
            continue;
        }

        bool isVeil = false;
        for (int j = 1; j < neighbourIndices.size(); j++)
        {
            int neighbourIndex = neighbourIndices[j];
            pcl::PointXYZI neighbourProjPoint = m_projectedCloud->points[neighbourIndex];
            float neighbourLength = lengthList[neighbourIndex];

            if (length - neighbourLength >= m_veilDistanceThreshold)
            {
                isVeil = true;
                break;
            }
        }

        if (isVeil)
        {
            m_veilPoints->push_back(point);
        }
        else
        {
            // 这是个真正的边界点
            m_boundaryPoints->push_back(point);
        }
    }
    m_boundaryPoints->width = m_boundaryPoints->points.size();
    m_boundaryPoints->height = 1;
    m_boundaryPoints->is_dense = true;
    m_veilPoints->width = m_veilPoints->points.size();
    m_veilPoints->height = 1;
    m_veilPoints->is_dense = true;
    m_borderPoints->width = m_borderPoints->points.size();
    m_borderPoints->height = 1;
    m_borderPoints->is_dense = true;
    qDebug() << "boundary points:" << m_boundaryPoints->size() << ", veil points:" << m_veilPoints->size() 
        << ", border points:" << m_borderPoints->size() << ", isolates points:" << isolates;
    TOCK("Classify_Boundary_Points");
}

void BoundaryExtractor::classifyBoundaryPoints2()
{
    qDebug() << "classifyBoundaryPoints2";
    m_boundaryPoints.reset(new pcl::PointCloud<pcl::PointXYZI>);
    m_veilPoints.reset(new pcl::PointCloud<pcl::PointXYZI>);
    m_borderPoints.reset(new pcl::PointCloud<pcl::PointXYZI>);
    m_boundaryMat = cv::Mat(m_matHeight, m_matWidth, CV_32F, cv::Scalar(0));
    cv::Mat indices(m_matHeight, m_matWidth, CV_32S, cv::Scalar(-1));

    for (int i = 0; i < m_allBoundary->points.size(); i++)
    {
        pcl::PointXYZI point = m_allBoundary->points[i];

        float z = point.z;
        // 按照从深度图计算点云的公式反算回去
        int pxX = static_cast<int>(point.x * m_fx / z + m_cx);
        int pxY = static_cast<int>(point.y * m_fy / z + m_cy);

        // 在生成边界点的深度图片时一并判断当前点是否为图像边缘点，如果是图像边缘点，
        // 则直接放入图像边缘点的分类点云中。这样在进行Veil点判断时，不必再次考虑像
        // 素坐标是否超出图像边缘的问题。
        if (pxX <= m_borderLeft || pxY <= m_borderTop || pxX >= (m_matWidth - m_borderRight) || pxY >= (m_matHeight - m_borderBottom))
        {
            // 在图像边缘的即为图像边框点
            m_borderPoints->push_back(point);
            continue;
        }
        // 将计算出的像素值加入到深度图中
        m_boundaryMat.row(pxY).at<float>(pxX) = z;
        // 同时将该像素对应的三维点序号也保存起来
        indices.row(pxY).at<float>(pxX) = i;
    }

    Eigen::Vector2f original(m_cx, m_cy);
    //qDebug() << "original:" << original.x() << original.y();

    int count = 0;
    for (int r = 0; r < m_boundaryMat.rows; r++)
    {
        for (int c = 0; c < m_boundaryMat.cols; c++)
        {
            int pointIndex = static_cast<int>(indices.row(r).at<float>(c));
            if (pointIndex < 0)
                continue;

            pcl::PointXYZI point = m_allBoundary->points[pointIndex];

            // 当前点作为检查中心点
            Eigen::Vector2f coord(c, r);

            // 从当前点到图像中心点建立二维射线
            Eigen::Vector2f ray = (original - coord).normalized();

            // 从检查的中心点，沿着射线方向，将一个光标一次向前移动一个单位向量的距离，然后计算光标位置周边的4个像素点是否为遮挡点，
            // 只要发现有遮挡点，则当前点即为被遮挡点，即为Veil点。否则当前点为有效的边界点。
            QList<cv::Point2i> processed;
            bool veil = false;
            for (int i = 1; i < m_classifyRadius; i++)
            {
                Eigen::Vector2f cursor = coord + ray * i;
                // 检查这周围的4个像素
                //cv::Point2i adj[4];
                int cursorX = qFloor(cursor.x());
                int cursorY = qFloor(cursor.y());

                /*adj[0] = cv::Point2i(qFloor(cursor.x()), qFloor(cursor.y()));
                adj[1] = cv::Point2i(qCeil(cursor.x()), qFloor(cursor.y()));
                adj[2] = cv::Point2i(qFloor(cursor.x()), qCeil(cursor.y()));
                adj[3] = cv::Point2i(qCeil(cursor.x()), qCeil(cursor.y()));*/
                for (int a = cursorY - 1; a <= cursorY + 1; a++)
                {
                    for (int b = cursorX - 1; b <= cursorX + 1; b++)
                    {
                        cv::Point2i block(b, a);

                        // 判断是否已经检查了当前光标下的点，防止重复检查
                        if (processed.contains(block))
                            continue;
                        // 把已经检查过的点放到processed集合中
                        processed.append(block);

                        // 获取像素点
                        int nPtIndex = static_cast<int>(indices.row(a).at<float>(b));
                        if (nPtIndex < 0)
                            continue;

                        // 获取光标下像素点对应的三维点
                        pcl::PointXYZI nPt = m_allBoundary->points[nPtIndex];

                        // 计算光标下像素点所对应的像素点与当前中心点的三维距离，如果该距离过小，则有可能是在同一个边界上，
                        // 这个距离要大于一个阈值，我们才认为是一个有效的遮挡点。
                        float diff = (point.getVector3fMap() - nPt.getVector3fMap()).norm();
                        if ((point.z - nPt.z) >= m_veilDistanceThreshold)
                        {
                            //qDebug().nospace().noquote() << "[" << c << ", " << r << "] -- [" << edgeX << ", " << edgeY << "] "
                                //<< i << ": [" << adj[a].x << ", " << adj[a].y << "] diff = " << diff;
                            veil = true;
                            break;
                        }
                    }
                    if (veil)
                        break;
                }
                if (veil)
                    break;
            }
            if (veil)
            {
                // 阴影点
                m_veilPoints->points.push_back(point);
            }
            else
            {
                // 边界点
                m_boundaryPoints->points.push_back(point);
            }
            count++;
        }
    }
    m_boundaryPoints->width = m_boundaryPoints->points.size();
    m_boundaryPoints->height = 1;
    m_boundaryPoints->is_dense = true;
    m_veilPoints->width = m_veilPoints->points.size();
    m_veilPoints->height = 1;
    m_veilPoints->is_dense = true;
    m_borderPoints->width = m_borderPoints->points.size();
    m_borderPoints->height = 1;
    m_borderPoints->is_dense = true;
    qDebug() << "boundary points:" << m_boundaryPoints->size() << ", veil points:" << m_veilPoints->size() << ", border points:" << m_borderPoints->size();
    qDebug() << "End classifyBoundaryPoints2";
}

void BoundaryExtractor::computeNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud)
{
    TICK("Compute_Normals");
    m_normals.reset(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    ne.setInputCloud(cloud);
    ne.setRadiusSearch(m_normalsRadiusSearch);
    ne.compute(*m_normals);
    qDebug() << "End computeNormals";
    TOCK("Compute_Normals");
}

void BoundaryExtractor::extractPlanes()
{
    m_planes.clear();

    // 先用欧式聚集，对点云进行分割。
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(m_filteredCloud);
    std::vector<pcl::PointIndices> clusterIndices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
    ec.setClusterTolerance(0.015); 
    ec.setMinClusterSize(m_filteredCloud->size() * 0.05);
    ec.setMaxClusterSize(m_filteredCloud->size());
    ec.setSearchMethod(tree);
    ec.setInputCloud(m_filteredCloud);
    ec.extract(clusterIndices);

    // 对分割出的点云再提取平面。因为使用SAC分割平面时，它不检测点是否连续，所以如果不先对点云分割就会取到非连续的平面。
    // 容易引入更大的误差。
    qDebug() << "cluster size:" << clusterIndices.size();
    for (int i = 0; i < clusterIndices.size(); i++)
    {
        // 参考以下两篇PCL官网教程
        // http://pointclouds.org/documentation/tutorials/extract_indices.php#extract-indices
        // http://pointclouds.org/documentation/tutorials/planar_segmentation.php
        qDebug() << "cluster" << i << ":" << clusterIndices[i].indices.size();
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud2(new pcl::PointCloud<pcl::PointXYZ>);
        pcl::copyPointCloud(*m_filteredCloud, clusterIndices[i].indices, *cloud);

        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        // Create the segmentation object
        pcl::SACSegmentation<pcl::PointXYZ> seg;
        // Optional
        seg.setOptimizeCoefficients(true);
        // Mandatory
        seg.setModelType(pcl::SACMODEL_PLANE);
        seg.setMethodType(pcl::SAC_RANSAC);
        seg.setDistanceThreshold(m_planeDistanceThreshold);

        pcl::ExtractIndices<pcl::PointXYZ> extract;
        while (cloud->points.size() >= 0.01f * m_filteredCloud->size())
        {
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
            seg.setInputCloud(cloud);
            seg.segment(*inliers, *coefficients);

            Plane plane;
            plane.parameters = coefficients;
            plane.cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);

            Eigen::Vector3f center(Eigen::Vector3f::Zero());
            for (int i = 0; i < inliers->indices.size(); i++)
            {
                pcl::PointXYZ point = cloud->points[inliers->indices[i]];
                center += point.getVector3fMap();
                plane.cloud->points.push_back(point);
            }
            center /= plane.cloud->points.size();
            Eigen::Vector3f normal = Eigen::Vector3f(coefficients->values[0], coefficients->values[1], coefficients->values[2]);
            float d = coefficients->values[3];
            plane.dir = normal.normalized();
            float dist = (center.dot(normal) + d) / normal.norm();
            plane.point = center - plane.dir * dist;
            plane.cloud->width = plane.cloud->points.size();
            plane.cloud->height = 1;
            plane.cloud->is_dense = true;
            plane.weight = static_cast<float>(inliers->indices.size()) / m_filteredCloud->size();

            extract.setInputCloud(cloud);
            extract.setIndices(inliers);
            extract.setNegative(false);
            extract.filter(*cloud2);

            extract.setNegative(true);
            extract.filter(*cloud2);
            cloud.swap(cloud2);

            // 检测一下当前平面的点数比例是否超过阈值，不超过的就抛弃。
            if (plane.weight > m_planePointsRate)
            {
                m_planes.append(plane);
            }
            qDebug() << "inliers:" << inliers->indices.size() << plane.weight << (plane.weight > 0.05f);
        }
    }
}
