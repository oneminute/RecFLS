#ifndef BOUNDARYEXTRACTOR_H
#define BOUNDARYEXTRACTOR_H

#include <QObject>

#include <pcl/common/common.h>
#include <pcl/features/eigen.h>
#include <pcl/search/kdtree.h>
#include <pcl/ModelCoefficients.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>

struct Plane
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Plane()
        : parameters(nullptr)
        , weight(0)
    {}

    pcl::ModelCoefficients::Ptr parameters;
    float weight;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
    Eigen::Vector3f point;
    Eigen::Vector3f dir;
};

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

    void boundaryEstimation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr gaussianFilter(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr outlierRemoval(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

    pcl::PointCloud<pcl::PointXYZ>::Ptr downSampling(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

    void classifyBoundaryPoints();

    void classifyBoundaryPoints2();

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

    cv::Mat boundaryMat() const { return m_boundaryMat; }

    pcl::PointCloud<pcl::PointXYZI>::Ptr projectedCloud() const { return m_projectedCloud; }
    pcl::PointCloud<pcl::PointXYZI>::Ptr boundaryPoints() const { return m_boundaryPoints; }
    pcl::PointCloud<pcl::PointXYZI>::Ptr veilPoints() const { return m_veilPoints; }
    pcl::PointCloud<pcl::PointXYZI>::Ptr borderPoints() const { return m_borderPoints; }

    QList<Plane> planes() { return m_planes; }

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

    float borderLeft() const { return m_borderLeft; }
    void setBorderLeft(float _borderLeft) { m_borderLeft = _borderLeft; }

    float borderRight() const { return m_borderRight; }
    void setBorderRight(float _borderRight) { m_borderRight = _borderRight; }

    float borderTop() const { return m_borderTop; }
    void setBorderTop(float _borderTop) { m_borderTop = _borderTop; }

    float borderBottom() const { return m_borderBottom; }
    void setBorderBottom(float _borderBottom) { m_borderBottom = _borderBottom; }

    float veilDistanceThreshold() const { return m_veilDistanceThreshold; }
    void setVeilDistanceThreshold(float _veilDistanceThreshold) { m_veilDistanceThreshold = _veilDistanceThreshold; }

    int downsamplingMethod() const { return static_cast<int>(m_downsamplingMethod); }
    void setDownsamplingMethod(int _downsamplingMethod) { m_downsamplingMethod = static_cast<DOWNSAMPLING_METHOD>(_downsamplingMethod); }

    bool enableRemovalFilter() const { return m_enableRemovalFilter; }
    void setEnableRemovalFilter(bool _enable) { m_enableRemovalFilter = _enable; }

    float crossPointsRadiusSearch() const { return m_crossPointsRadiusSearch; }
    void setCrossPointsRadiusSearch(float _value) { m_crossPointsRadiusSearch = _value; }

    float crossPointsClusterTolerance() const { return m_crossPointsClusterTolerance; }
    void setCrossPointsClusterTolerance(float _value) { m_crossPointsClusterTolerance = _value; }

    float curvatureThreshold() const { return m_curvatureThreshold; }
    void setCurvatureThreshold(float _value) { m_curvatureThreshold = _value; }

    int minNormalClusters() const { return m_minNormalClusters; }
    void setMinNormalClusters(float _value) { m_minNormalClusters = _value; }

    int maxNormalClusters() const { return m_maxNormalClusters; }
    void setMaxNormalCulsters(float _value) { m_maxNormalClusters = _value; }

    float planeDistanceThreshold() const { return m_planeDistanceThreshold; }
    void setPlaneDistanceThreshold(float _value) { m_planeDistanceThreshold = _value; }

    float planePointsRate() const { return m_planePointsRate; }
    void setPlanePointsRate(float _value) { m_planePointsRate = _value; }

protected:
    void computeNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

    void extractPlanes();

private:
    // 输入点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_cloud;
    //pcl::IndicesPtr m_indices;

    // 下采样后的点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_downsampledCloud;

    // 高斯过滤后的点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_filteredCloud;

    // 离群点移除后的点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_removalCloud;

    // 法线
    pcl::PointCloud<pcl::Normal>::Ptr m_normals;

    // 抽取出的所有边界点
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_allBoundary;

    // 投影点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_projectedCloud;

    // 用于分类边界点的深度图片，该图片由边界点反向投影到二维深度图上
    cv::Mat m_boundaryMat;

    // 真正的边界图片
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_boundaryPoints;

    // 阴影图片
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_veilPoints;

    // 屏幕边缘图片
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_borderPoints;

    // 抽取出的平面
    QList<Plane> m_planes;

    DOWNSAMPLING_METHOD m_downsamplingMethod;
    bool m_enableRemovalFilter;
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
    float m_borderLeft;
    float m_borderRight;
    float m_borderTop;
    float m_borderBottom;
    float m_projectedRadiusSearch;
    float m_veilDistanceThreshold;
    float m_crossPointsRadiusSearch;
    float m_crossPointsClusterTolerance;
    float m_curvatureThreshold;
    int m_minNormalClusters;
    int m_maxNormalClusters;
    float m_planeDistanceThreshold;

    int m_classifyRadius;
    float m_planePointsRate;
};


#endif // BOUNDARYEXTRACTOR_H
