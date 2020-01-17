#ifndef BOUNDARYEXTRACTOR_H
#define BOUNDARYEXTRACTOR_H

#include <QObject>

#include <pcl/common/common.h>
#include <pcl/features/eigen.h>
#include <pcl/search/kdtree.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>


class BoundaryExtractor : public QObject
{
    Q_OBJECT
public:
    explicit BoundaryExtractor(QObject* parent = nullptr);

    pcl::PointCloud<pcl::PointXYZI>::Ptr compute();

    void setInputCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& _cloud)
    {
        m_cloud = _cloud;
    }

    /*void setIndices(const pcl::IndicesPtr& _indices)
    {
        m_indices = _indices;
    }

    pcl::IndicesPtr indices() const { return m_indices; }*/

    void setNormals(const pcl::PointCloud<pcl::Normal>::Ptr& _normals)
    {
        m_normals = _normals;
    }

    pcl::PointCloud<pcl::Normal>::Ptr normals() const { return m_normals; }

    pcl::PointCloud<pcl::PointXYZ>::Ptr filteredCloud() const { return m_filteredCloud; }

    int OutlierRemovalMeanK() const { return m_outlierRemovalMeanK; }
    void setOutlierRemovalMeanK(int _outlierRemovalMeanK) { m_outlierRemovalMeanK = _outlierRemovalMeanK; }

    float StddevMulThresh() const { return m_stddevMulThresh; }
    void setStddevMulThresh(float _stddevMulThresh) { m_stddevMulThresh = _stddevMulThresh; }

    float gaussianSigma() const { return m_gaussianSigma; }
    void setGaussianSigma(float _gaussianSigma) { m_gaussianSigma = _gaussianSigma; }

    float gaussianRSigma() const { return m_gaussianRSigma; }
    void setGaussianRSigma(float _gaussianRSigma) { m_gaussianRSigma = _gaussianRSigma; }

    float gaussianRadiusSearch() const { return m_gaussianRadiusSearch; }
    void setGaussianRadiusSearch(float _gaussianRadiusSearch) { m_gaussianRadiusSearch = _gaussianRadiusSearch; }

    float downsampleLeafSize() const { return m_downsampleLeafSize; }
    void setDownsampleLeafSize(float _downsampleLeafSize) { m_downsampleLeafSize = _downsampleLeafSize; }

    float normalsRadiusSearch() const { return m_normalsRadiusSearch; }
    void setNormalsRadiusSearch(float _normalsRadiusSearch) { m_normalsRadiusSearch = _normalsRadiusSearch; }

    float boundaryRadiusSearch() const { return m_boundaryRadiusSearch; }
    void setBoundaryRadiusSearch(float _boundaryRadiusSearch) { m_boundaryRadiusSearch = _boundaryRadiusSearch; }

    float boundaryAngleThreshold() const { return m_boundaryAngleThreshold; }
    void setBoundaryAngleThreshold(float _boundaryAngleThreshold) { m_boundaryAngleThreshold = _boundaryAngleThreshold; }

protected:
    void computeNormals();

private:
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_cloud;
    //pcl::IndicesPtr m_indices;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_downsampledCloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_filteredCloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_removalCloud;
    pcl::PointCloud<pcl::Normal>::Ptr m_normals;
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_boundary;

    float m_downsampleLeafSize;
    int m_outlierRemovalMeanK;
    float m_stddevMulThresh;
    float m_gaussianSigma;
    float m_gaussianRSigma;
    float m_gaussianRadiusSearch;
    float m_normalsRadiusSearch;
    float m_boundaryRadiusSearch;
    float m_boundaryAngleThreshold;
};


#endif // BOUNDARYEXTRACTOR_H
