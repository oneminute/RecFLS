#include "BoundaryExtractor.h"

#include <QDebug>

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
    , m_downsampleLeafSize(0.005f)
    , m_outlierRemovalMeanK(50)
    , m_stddevMulThresh(1.0f)
    , m_gaussianSigma(4)
    , m_gaussianRSigma(4)
    , m_gaussianRadiusSearch(0.05f)
    , m_normalsRadiusSearch(0.05f)
    , m_boundaryRadiusSearch(0.1f)
    , m_boundaryAngleThreshold(M_PI_4)
{

}

pcl::PointCloud<pcl::PointXYZI>::Ptr BoundaryExtractor::compute()
{
    m_boundary.reset(new pcl::PointCloud<pcl::PointXYZI>);
    m_downsampledCloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    m_filteredCloud.reset(new pcl::PointCloud<pcl::PointXYZ>);
    m_removalCloud.reset(new pcl::PointCloud<pcl::PointXYZ>);

    qDebug() << "downsampleLeafSize:" << m_downsampleLeafSize;

    if (!m_cloud || m_cloud->empty())
        return m_boundary;

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
//    pcl::UniformSampling<pcl::PointXYZ> us;
//    us.setIndices(m_indices);
//    us.setInputCloud(m_cloud);
//    us.setRadiusSearch(m_downsampleLeafSize);
//    us.filter(*m_downsampledCloud);

    //pcl::PCLPointCloud2::Ptr cloud2(new pcl::PCLPointCloud2);
    //pcl::toPCLPointCloud2(*m_cloud, *cloud2);

    pcl::VoxelGrid<pcl::PointXYZ> vg;
    //vg.setIndices(m_indices);
    vg.setInputCloud(m_cloud);
    vg.setLeafSize(m_downsampleLeafSize, m_downsampleLeafSize, m_downsampleLeafSize);
    vg.filter(*m_downsampledCloud);

    //pcl::fromPCLPointCloud2(*cloud2, *m_downsampledCloud);

    qDebug() << "Size before downsampling:" << m_cloud->size() << ", size after:" << m_downsampledCloud->size();

    //m_indices->clear();
    /*for (int i = 0; i < m_downsampledCloud->size(); i++)
    {
        m_indices->push_back(i);
    }*/
    TOCK("Downsampling");

    TICK("Outlier_Removal");
    // 剔除离群点
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
            pt.intensity = i;
            m_boundary->points.push_back(pt);
        }
    }
    m_boundary->width = m_boundary->points.size();
    m_boundary->height = 1;
    m_boundary->is_dense = true;
    qDebug() << "Extract boundary size:" << m_boundary->size() << ", from original cloud size:" << m_cloud->size();
    TOCK("Boundary_Estimation");

    return m_boundary;
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
