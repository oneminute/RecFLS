#include "BoundaryExtractor.h"

#include <QDebug>
#include <QtMath>

#include <pcl/common/common.h>
#include <pcl/features/boundary.h>
#include <pcl/features/impl/normal_3d.hpp>
#include <pcl/features/normal_3d.h>
#include <pcl/search/kdtree.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/convolution_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/uniform_sampling.h>

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
    , m_cx(320)
    , m_cy(240)
    , m_fx(583)
    , m_fy(583)
    , m_borderWidth(26)
    , m_borderHeight(16)
    , m_projectedRadiusSearch(M_PI / 72)
    , m_veilDistanceThreshold(0.1f)
{

}

pcl::PointCloud<pcl::PointXYZI>::Ptr BoundaryExtractor::compute()
{
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

    pcl::Indices indices;

    TICK("Downsampling");
    if (m_downsamplingMethod == DM_VOXEL_GRID)
    {
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        //vg.setIndices(m_indices);
        vg.setInputCloud(m_cloud);
        vg.setLeafSize(m_downsampleLeafSize, m_downsampleLeafSize, m_downsampleLeafSize);
        vg.filter(*m_downsampledCloud);
    }
    if (m_downsamplingMethod == DM_UNIFORM_SAMPLING)
    {
        pcl::UniformSampling<pcl::PointXYZ> us;
        //us.setIndices(m_indices);
        us.setInputCloud(m_cloud);
        us.setRadiusSearch(m_downsampleLeafSize);
        us.filter(*m_downsampledCloud);
    }
    
    qDebug() << "Size before downsampling:" << m_cloud->size() << ", size after:" << m_downsampledCloud->size();

    //m_indices->clear();
    /*for (int i = 0; i < m_downsampledCloud->size(); i++)
    {
        m_indices->push_back(i);
    }*/
    TOCK("Downsampling");

    TICK("Outlier_Removal");
    // 剔除离群点
    if (m_enableRemovalFilter)
    {
        pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
        //sor.setIndices(m_indices);
        sor.setInputCloud(m_downsampledCloud);
        sor.setMeanK(m_outlierRemovalMeanK);
        sor.setStddevMulThresh(m_stddevMulThresh); //距离大于1倍标准方差
        sor.filter(indices);
        //m_indices.reset(new std::vector<int>(indices));
        pcl::copyPointCloud(*m_downsampledCloud, indices, *m_removalCloud);
        qDebug() << "Cloud size before outlier removal:" << m_downsampledCloud->size() << ", size after:" << m_removalCloud->size() << ", removed size:" << (m_downsampledCloud->size() - indices.size());
        TOCK("Outlier_Removal");
    }
    else
    {
        m_removalCloud = m_downsampledCloud;
    }

    TICK("Gaussian_Filter");
    // 高斯滤波
    pcl::filters::GaussianKernel<pcl::PointXYZ, pcl::PointXYZ>::Ptr kernel(new pcl::filters::GaussianKernel<pcl::PointXYZ, pcl::PointXYZ>);
    kernel->setSigma(m_gaussianSigma);
    kernel->setThresholdRelativeToSigma(m_gaussianRSigma);

    //Set up the Convolution Filter
    pcl::filters::Convolution3D<pcl::PointXYZ, pcl::PointXYZ, pcl::filters::GaussianKernel<pcl::PointXYZ, pcl::PointXYZ>> convolution;
    convolution.setKernel(*kernel);
    convolution.setInputCloud(m_removalCloud);
    //convolution.setIndices(m_indices);
    convolution.setRadiusSearch(m_gaussianRadiusSearch);
    convolution.convolve(*m_filteredCloud);
    TOCK("Gaussian_Filter");

    TICK("Compute_Normals");
    if (!m_normals || m_normals->empty())
    {
        computeNormals();
    }
    TOCK("Compute_Normals");

    TICK("Boundary_Estimation");
    //calculate boundary_;
    pcl::PointCloud<pcl::Boundary> boundary;
    pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> be;
    be.setInputCloud(m_filteredCloud);
    //be.setIndices(m_indices);
    be.setInputNormals(m_normals);
    be.setRadiusSearch(m_boundaryRadiusSearch);
    be.setAngleThreshold(m_boundaryAngleThreshold);
    //be.setSearchMethod(tree);
    be.compute(boundary);

    for (int i = 0; i < boundary.points.size(); i++)
    {
        if (boundary[i].boundary_point == 1)
        {
            pcl::PointXYZI pt;
            pt.x = m_filteredCloud->points[i].x;
            pt.y = m_filteredCloud->points[i].y;
            pt.z = m_filteredCloud->points[i].z;
            pt.intensity = 0;
            m_allBoundary->points.push_back(pt);
        }
    }
    m_allBoundary->width = m_allBoundary->points.size();
    m_allBoundary->height = 1;
    m_allBoundary->is_dense = true;
    qDebug() << "Extract boundary size:" << m_allBoundary->size() << ", from original cloud size:" << m_cloud->size();
    TOCK("Boundary_Estimation");

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

        int i = static_cast<int>(point.x * m_fx / point.z + m_cx);
        int j = static_cast<int>(point.y * m_fy / point.z + m_cy);
        
        if (i <= m_borderWidth || j <= m_borderHeight || i >= (m_matWidth - m_borderWidth) || j >= (m_matHeight - m_borderHeight))
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

        pcl::Indices neighbourIndices;
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

    return m_allBoundary;
}

void BoundaryExtractor::computeNormals()
{
    m_normals.reset(new pcl::PointCloud<pcl::Normal>);
    pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
    //ne.setIndices(m_indices);
    ne.setInputCloud(m_filteredCloud);
    ne.setRadiusSearch(m_normalsRadiusSearch);
    ne.compute(*m_normals);
}
