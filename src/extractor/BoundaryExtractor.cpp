#include "BoundaryExtractor.h"

#include <QDebug>
#include <QtMath>

#include <pcl/common/common.h>
#include <pcl/features/boundary.h>
#include <pcl/features/impl/normal_3d.hpp>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/convolution_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/search.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/impl/extract_clusters.hpp>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/sac_segmentation.h>

#include "util/StopWatch.h"

BoundaryExtractor::BoundaryExtractor(QObject* parent)
    : QObject(parent)
    /*, m_downsamplingMethod(DM_VOXEL_GRID)
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
    , m_projectedRadiusSearch(M_PI / 72)
    , m_veilDistanceThreshold(0.075f)
    , m_crossPointsRadiusSearch(0.05f)
    , m_crossPointsClusterTolerance(0.1f)
    , m_curvatureThreshold(0.025f)
    , m_minNormalClusters(2)
    , m_maxNormalClusters(2)
    , m_planeDistanceThreshold(0.01)
    , m_classifyRadius(20)
    , m_planePointsRate(0.05f)*/
    , PROPERTY_INIT(Cx, 320)
    , PROPERTY_INIT(Cy, 240)
    , PROPERTY_INIT(Fx, 583)
    , PROPERTY_INIT(Fy, 583)
    , PROPERTY_INIT(Width, 640)
    , PROPERTY_INIT(Height, 480)
    , PROPERTY_INIT(BorderLeft, 26)
    , PROPERTY_INIT(BorderRight, 8)
    , PROPERTY_INIT(BorderTop, 4)
    , PROPERTY_INIT(BorderBottom, 4)
    , PROPERTY_INIT(DepthShift, 1000.f)
    , PROPERTY_INIT(MinDepth, 0.4f)
    , PROPERTY_INIT(MaxDepth, 8.0f)
    , PROPERTY_INIT(CudaNormalKernalRadius, 20)
    , PROPERTY_INIT(CudaNormalKnnRadius, 0.1f)
    , PROPERTY_INIT(CudaBEDistance, 0.1f)
    , PROPERTY_INIT(CudaBEAngleThreshold, 45)
    , PROPERTY_INIT(CudaBEKernalRadius, 20)
    , PROPERTY_INIT(CudaGaussianSigma, 10.f)
    , PROPERTY_INIT(CudaGaussianKernalRadius, 20)
    , PROPERTY_INIT(CudaClassifyKernalRadius, 20)
    , PROPERTY_INIT(CudaClassifyDistance, 0.2f)
    , PROPERTY_INIT(VBRGResolution, 0.1)
    , PROPERTY_INIT(VBRGMinPoints, 10)
    , PROPERTY_INIT(CudaPeakClusterTolerance, 5)
    , PROPERTY_INIT(CudaMinClusterPeaks, 2)
    , PROPERTY_INIT(CudaMaxClusterPeaks, 3)
    , PROPERTY_INIT(CudaCornerHistSigma, 1.0f)
{

}

pcl::PointCloud<pcl::PointXYZI>::Ptr BoundaryExtractor::computeCUDA(cuda::GpuFrame& frame)
{
    frame.parameters.cx = Cx();
    frame.parameters.cy = Cy();
    frame.parameters.fx = Fx();
    frame.parameters.fy = Fy();
    frame.parameters.minDepth = MinDepth();
    frame.parameters.maxDepth = MaxDepth();
    frame.parameters.borderLeft = BorderLeft();
    frame.parameters.borderRight = BorderRight();
    frame.parameters.borderTop = BorderTop();
    frame.parameters.borderBottom = BorderBottom();
    frame.parameters.depthShift = DepthShift();
    frame.parameters.normalKernalRadius = CudaNormalKernalRadius();
    frame.parameters.normalKnnRadius = CudaNormalKnnRadius();
    frame.parameters.boundaryEstimationRadius = CudaBEKernalRadius();
    frame.parameters.boundaryGaussianSigma = CudaGaussianSigma();
    frame.parameters.boundaryGaussianRadius = CudaGaussianKernalRadius();
    frame.parameters.boundaryEstimationDistance = CudaBEDistance();
    frame.parameters.boundaryAngleThreshold = CudaBEAngleThreshold();
    frame.parameters.classifyRadius = CudaClassifyKernalRadius();
    frame.parameters.classifyDistance = CudaClassifyDistance();
    frame.parameters.peakClusterTolerance = CudaPeakClusterTolerance();
    frame.parameters.minClusterPeaks = CudaMinClusterPeaks();
    frame.parameters.maxClusterPeaks = CudaMaxClusterPeaks();
    frame.parameters.cornerHistSigma = CudaCornerHistSigma();

    cv::cuda::GpuMat boundaryMatGpu(Height(), Width(), CV_8U, frame.boundaryImage);
    cv::cuda::GpuMat pointsMatGpu(Height(), Width(), CV_32S, frame.indicesImage);

    TICK("extracting_boundaries");
    cuda::generatePointCloud(frame);
    TOCK("extracting_boundaries");

    TICK("boundaries_downloading");
    boundaryMatGpu.download(m_boundaryMat);
    pointsMatGpu.download(m_pointsMat);
    std::vector<float3> points;
    frame.pointCloud.download(points);
    std::vector<float3> normals;
    frame.pointCloudNormals.download(normals);
    std::vector<uchar> boundaries;
    frame.boundaries.download(boundaries);

    m_cloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    m_allBoundary.reset(new pcl::PointCloud<pcl::PointXYZI>);
    m_boundaryPoints.reset(new pcl::PointCloud<pcl::PointXYZI>);
    m_cornerPoints.reset(new pcl::PointCloud<pcl::PointXYZI>);
    m_veilPoints.reset(new pcl::PointCloud<pcl::PointXYZI>);
    m_borderPoints.reset(new pcl::PointCloud<pcl::PointXYZI>);
    m_normals.reset(new pcl::PointCloud<pcl::Normal>);
    int negativeNum = 0;
    for(int i = 0; i < Height(); i++) 
    {
        for(int j = 0; j < Width(); j++) 
        {
            cv::Point coord(j, i);
            int index = i * Width() + j;
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
                else if (pointType == 4)
                {
                    m_cornerPoints->points.push_back(ptI);
                }
                m_indices.push_back(index);
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
    qDebug() << "all boundary points:" << m_allBoundary->size() << ", boundary points:" << m_boundaryPoints->size() 
        << ", veil points:" << m_veilPoints->size() << ", border points:" << m_borderPoints->size() << ", corner points:" << m_cornerPoints->size();
    TOCK("boundaries_downloading");

    return m_allBoundary;
}

pcl::PointCloud<pcl::PointXYZI>::Ptr BoundaryExtractor::computeVBRG()
{
    float resolution = VBRGResolution();
    int minPoints = VBRGMinPoints();

    m_normals.reset(new pcl::PointCloud<pcl::Normal>);
    m_voxelPoints.reset(new pcl::PointCloud<pcl::PointXYZ>);
    m_voxelInfos.clear();

    pcl::PointCloud<pcl::PointXYZI>::Ptr classifiedCloud(new pcl::PointCloud<pcl::PointXYZI>);

    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(resolution);
    octree.setInputCloud(m_cloud);
    octree.addPointsFromInputCloud();

    double minX, minY, minZ, maxX, maxY, maxZ;
    octree.getBoundingBox(minX, minY, minZ, maxX, maxY, maxZ);
    int leafCount = octree.getLeafCount();
    int branchCount = octree.getBranchCount();

    qDebug() << "[BoundaryExtractor::computeVBRG] branchCount:" << branchCount << ", leafCount:" << leafCount;
    qDebug() << "[BoundaryExtractor::computeVBRG] bounding box:" << minX << minY << minZ << maxX << maxY << maxZ;

    pcl::octree::OctreePointCloud<pcl::PointXYZ>::LeafNodeIterator it(&octree);
    while (it != octree.leaf_end())
    {
        //qDebug() << it.getNodeID() << ", branch:" << it.isBranchNode() << ", leaf:" << it.isLeafNode() << ", depth:" << it.getCurrentOctreeDepth() << ", points size:" << it.getLeafContainer().getSize();
        VoxelInfo vi;
        vi.nodeId = it.getNodeID();
        vi.sideLength = resolution;
        vi.state = 1;
        
        std::vector<int> indices = it.getLeafContainer().getPointIndicesVector();
        //vi.center = m_cloud->points[indices[indices.size() / 2]].getVector3fMap();

        vi.center = Eigen::Vector3f::Zero();
        for (int i = 0; i < indices.size(); i++)
        {
            pcl::PointXYZ point = m_cloud->points[indices[i]];
            vi.center += point.getVector3fMap();  // 所有点累加
        }
        vi.center /= indices.size();      // 计算样本均值

        std::vector<float> dists;
        pcl::PointXYZ centerPt;
        centerPt.getVector3fMap() = vi.center;
        octree.radiusSearch(centerPt, qSqrt(octree.getVoxelSquaredDiameter()) / 2, indices, dists, m_cloud->size());

        consistencyOptimization(m_cloud, indices, vi, minPoints);

        if (vi.state == 1)
        {
            // 需要向上合并节点
        }
        else if (vi.state == 2)
        {
            // 需要向下切分节点
        }
        else
        {
            // 特征描述符，三个分量分别表示a1d a2d a3c
            vi.axd.x() = (qSqrt(vi.lambdas.x()) - qSqrt(vi.lambdas.y())) / qSqrt(vi.lambdas.x());
            vi.axd.y() = (qSqrt(vi.lambdas.y()) - qSqrt(vi.lambdas.z())) / qSqrt(vi.lambdas.x());
            vi.axd.z() = qSqrt(vi.lambdas.z()) / qSqrt(vi.lambdas.x());
            vi.axd.maxCoeff(&vi.axdIndex);
            vi.ef = -vi.axd.x() * qLn(vi.axd.x()) - vi.axd.y() * qLn(vi.axd.y()) - vi.axd.z() * qLn(vi.axd.z());   // 香农熵，表示该点集包含的信息量
            vi.stdDiv = qSqrt(vi.axd.z());        // 拟合标准差

            //std::cout << "    " << vi.center.transpose() << std::endl;
            pcl::Normal normal;
            normal.getNormalVector3fMap() = vi.abc;    // 该向量即为abc参数，也即为法线向量，a^2 + b^2 + c^2 = 1
            pcl::PointXYZ voxelPoint;
            voxelPoint.getVector3fMap() = vi.center;      // 样本均值设为该八叉树叶结点的质点

            indices = it.getLeafContainer().getPointIndicesVector();
            m_normals->points.push_back(normal);
            m_voxelPoints->points.push_back(voxelPoint);
        }

        Eigen::Vector3f bMin, bMax;
        octree.getVoxelBounds(it, bMin, bMax);
        float diameter = qSqrt(octree.getVoxelSquaredDiameter());
        float sideLength = octree.getVoxelSquaredSideLen();

        // 计算当前块的坐标，整数值
        vi.min = bMin;
        vi.max = bMax;
        m_voxelInfos.insert(vi.nodeId, vi);

        //qDebug() << "    " << vi.pos.x() << vi.pos.y() << vi.pos.z();
        //qDebug() << "    " << vi.abc.x() << vi.abc.y() << vi.abc.z() << vi.d;
        //qDebug() << "    type:" << axdIndex << ", a1d =" << axd.x() << ", a2d =" << axd.y() << ", a3d =" << axd.z();
        //qDebug() << "    vi.ef:" << vi.ef;
        //qDebug() << "    stdDiv:" << vi.stdDiv;
        //qDebug() << "    deameter:" << diameter << ", sideLength:" << sideLength;
        it++;

        for (int i = 0; i < indices.size(); i++)
        {
            pcl::PointXYZ point = m_cloud->points[indices[i]];
            pcl::PointXYZI outPoint;
            outPoint.getVector3fMap() = point.getVector3fMap();
            outPoint.intensity = vi.axd.y();
            classifiedCloud->points.push_back(outPoint);
        }
    }
    m_voxelPoints->width = m_voxelPoints->points.size();
    m_voxelPoints->height = 1;
    m_voxelPoints->is_dense = true;
    m_normals->width = m_normals->points.size();
    m_normals->height = 1;
    m_normals->is_dense = true;
    classifiedCloud->width = classifiedCloud->points.size();
    classifiedCloud->height = 1;
    classifiedCloud->is_dense = true;

    return classifiedCloud;
}

void BoundaryExtractor::fitPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::vector<int>& indices, 
    Eigen::Vector3f& abc, float& d, Eigen::Vector3f& avgPoint, Eigen::Vector3f& lambdas)
{
    Eigen::Matrix3f axis;
    axis << 1, 0, 0,
        0, -1, 0,
        0, 0, 1;

    avgPoint = Eigen::Vector3f::Zero();
    for (int i = 0; i < indices.size(); i++)
    {
        pcl::PointXYZ point = cloud->points[indices[i]];
        avgPoint += point.getVector3fMap();  // 所有点累加
    }
    avgPoint /= indices.size();      // 计算样本均值
    
    //qDebug() << "    avg:" << avgPoint.x() << avgPoint.y() << avgPoint.z() << ", indices size:" << indices.size();
    Eigen::MatrixXf samples;    // 样本矩阵为3xn的矩阵中，每列为一个点向量
    samples.resize(3, indices.size());
    for (int i = 0; i < indices.size(); i++)
    {
        pcl::PointXYZ point = m_cloud->points[indices[i]];
        samples.col(i) = point.getVector3fMap() - avgPoint;      // 计算每个点与均值的差，放到样本矩阵中
    }
    Eigen::Matrix3f A = samples * samples.transpose();      // 生成3x3的协方差矩阵
    Eigen::EigenSolver<Eigen::Matrix3f> es(A);              // 解算特征值与特征向量
    Eigen::Vector3f eigenValues = es.eigenvalues().real();
    Eigen::Index minIndex, midIndex, maxIndex;
    eigenValues.minCoeff(&minIndex);        // 获得最小的特征值
    eigenValues.maxCoeff(&maxIndex);
    for (int i = 0; i < 3; i++)
    {
        if (i != minIndex && i != maxIndex)
        {
            midIndex = i;
            break;
        }
    }
    
    abc = es.eigenvectors().real().col(minIndex);   // 获得最小的特征向量
    // 法线方向一致性
    Eigen::Vector3f axisIndex;
    axisIndex.x() = qAbs(abc.x());
    axisIndex.y() = qAbs(abc.y());
    axisIndex.z() = qAbs(abc.z());
    Eigen::Index maxAxis;
    axisIndex.maxCoeff(&maxAxis);
    if (abc.dot(axis.col(maxAxis)) < 0)
    {
        abc = -abc;
    }
    d = (abc.x() * avgPoint.x() + abc.y() * avgPoint.y() + abc.z() * avgPoint.z()) / indices.size();

    // 按从大到小顺序排序特征值
    lambdas.x() = eigenValues[maxIndex];
    lambdas.y() = eigenValues[midIndex];
    lambdas.z() = eigenValues[minIndex];
}

void BoundaryExtractor::consistencyOptimization(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::vector<int>& indices, VoxelInfo& vi, int minPoints)
{
    int iteration = 0;
    while (indices.size() >= 3)
    {
        vi.state = 0;
        Eigen::Vector3f avgPoint;
        Eigen::Vector3f lambdas;
        fitPlane(m_cloud, indices, vi.abc, vi.d, avgPoint, lambdas);

        if (iteration == 0)
        {
            vi.center.x() = avgPoint.x();
            vi.center.y() = avgPoint.y();
            vi.center.z() = avgPoint.z();

            vi.lambdas.x() = lambdas.x();
            vi.lambdas.y() = lambdas.y();
            vi.lambdas.z() = lambdas.z();
        }

        std::vector<float> ds;          // 保存所有di的集合
        float avgD = 0;                     // di的平均值
        for (int i = 0; i < indices.size(); i++)
        {
            pcl::PointXYZ point = m_cloud->points[indices[i]];
            Eigen::Vector3f diff = point.getVector3fMap() - avgPoint;
            float di = qAbs(diff.dot(vi.abc));  // 计算每一个点到平面的距离
            ds.push_back(di);
            avgD += di;
        }
        avgD /= indices.size();     // 求均值

        int number = indices.size();
        std::vector<int> tmpIndices;
        for (int i = 0; i < ds.size(); i++)
        {
            float di = ds[i];
            float sigmai = qSqrt((di - avgD) * (di - avgD) / (ds.size() - 1)); //计算单点的标准偏差
            //qDebug() << "    di:" << di << ", sigmai:" << (sigmai * 10) << (di <= sigmai * 10);
            if (di <= sigmai * 10)
            {
                tmpIndices.push_back(indices[i]);
            }
        }
        indices = tmpIndices;

        //qDebug() << "    iteration:" << iteration << ", before size:" << number << ", after size:" << indices.size();
        iteration++;

        if (indices.size() < minPoints)
        {
            break;
        }

        vi.d = avgPoint.dot(vi.abc) / indices.size();

        if (number == indices.size())
        {
            // 没有要过滤掉的点了，都在误差范围内
            break;
        }
    }
}

