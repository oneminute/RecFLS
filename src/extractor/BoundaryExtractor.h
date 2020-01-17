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
    enum DOWNSAMPLING_METHOD
    {
        DM_VOXEL_GRID = 0,
        DM_UNIFORM_SAMPLING
    };

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

    pcl::PointCloud<pcl::PointXYZI>::Ptr projectedCloud() const { return m_projectedCloud; }
    pcl::PointCloud<pcl::PointXYZI>::Ptr boundaryPoints() const { return m_boundaryPoints; }
    pcl::PointCloud<pcl::PointXYZI>::Ptr veilPoints() const { return m_veilPoints; }
    pcl::PointCloud<pcl::PointXYZI>::Ptr borderPoints() const { return m_borderPoints; }

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

    float matWidth() const { return m_matWidth; }
    void setMatWidth(int _matWidth) { m_matWidth = _matWidth; }

    float matHeight() const { return m_matHeight; }
    void setMatHeight(int _matHeight) { m_matHeight = _matHeight; }

    float cx() const { return m_cx; }
    void setCx(float _cx) { m_cx = _cx; }

    float cy() const { return m_cy; }
    void setCy(float _cy) { m_cy = _cy; }

    float fx() const { return m_fx; }
    void setFx(float _fx) { m_fx = _fx; }

    float fy() const { return m_fy; }
    void setFy(float _fy) { m_fy = _fy; }

    float projectedRadiusSearch() const { return m_projectedRadiusSearch; }
    void setProjectedRadiusSearch(float _projectedRadiusSearch) { m_projectedRadiusSearch = _projectedRadiusSearch; }

    float borderWidth() const { return m_borderWidth; }
    void setBorderWidth(float _borderWidth) { m_borderWidth = _borderWidth; }

    float veilDistanceThreshold() const { return m_veilDistanceThreshold; }
    void setVeilDistanceThreshold(float _veilDistanceThreshold) { m_veilDistanceThreshold = _veilDistanceThreshold; }

protected:
    void computeNormals();

private:
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_cloud;
    //pcl::IndicesPtr m_indices;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_downsampledCloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_filteredCloud;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_removalCloud;
    pcl::PointCloud<pcl::Normal>::Ptr m_normals;
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_allBoundary;
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_projectedCloud;

    pcl::PointCloud<pcl::PointXYZI>::Ptr m_boundaryPoints;
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_veilPoints;
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_borderPoints;

    float m_downsampleLeafSize;
    int m_outlierRemovalMeanK;
    float m_stddevMulThresh;
    float m_gaussianSigma;
    float m_gaussianRSigma;
    float m_gaussianRadiusSearch;
    float m_normalsRadiusSearch;
    float m_boundaryRadiusSearch;
    float m_boundaryAngleThreshold;
    int m_matWidth;
    int m_matHeight;
    float m_cx;
    float m_cy;
    float m_fx;
    float m_fy;
    float m_projectedRadiusSearch;
    float m_borderWidth;
    float m_veilDistanceThreshold;
};


#endif // BOUNDARYEXTRACTOR_H
