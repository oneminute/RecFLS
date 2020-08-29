#ifndef BOUNDARYEXTRACTOR_H
#define BOUNDARYEXTRACTOR_H

#include <QObject>
#include <QMap>

#include <pcl/common/common.h>
#include <pcl/features/eigen.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/octree/octree.h>
#include <pcl/search/kdtree.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <opencv2/opencv.hpp>

#include "common/Common.h"
#include "cuda/CudaInternal.h"

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

struct VoxelInfo
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VoxelInfo()
        : state(0)
        , pointSize(0)
        , lambdas(Eigen::Vector3f::Zero())
        , axd(Eigen::Vector3f::Zero())
        , ef(0)
        , stdDiv(0)
        , axdIndex(0)
    {}
    qulonglong nodeId;
    int state;
    quint32 pointSize;
    float sideLength;
    Eigen::Vector3f center;
    Eigen::Vector3f min;
    Eigen::Vector3f max;
    Eigen::Vector3f abc;
    float d;
    Eigen::Vector3f lambdas;
    Eigen::Vector3f axd;
    Eigen::Index axdIndex;
    float ef;
    float stdDiv;
};

class BEOctree;

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

    pcl::PointCloud<pcl::PointXYZI>::Ptr computeCUDA(cuda::GpuFrame& frame);

    pcl::PointCloud<pcl::PointXYZI>::Ptr computeVBRG();

    void fitPlane(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::vector<int>& indices, Eigen::Vector3f& abc, float& d, Eigen::Vector3f& avgPoint, Eigen::Vector3f& lambdas);

    void consistencyOptimization(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::vector<int>& indices, VoxelInfo& vi, int minPoints);

    void boundaryEstimation(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);

    void setInputCloud(const pcl::PointCloud<pcl::PointXYZ>::Ptr& _cloud)
    {
        m_cloud = _cloud;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud() const { return m_cloud; }

    void setNormals(const pcl::PointCloud<pcl::Normal>::Ptr& _normals)
    {
        m_normals = _normals;
    }

    pcl::PointCloud<pcl::Normal>::Ptr normals() const { return m_normals; }

    pcl::PointCloud<pcl::PointXYZ>::Ptr filteredCloud() const { return m_filteredCloud; }

    cv::Mat boundaryMat() const { return m_boundaryMat; }

    cv::Mat pointsMat() const { return m_pointsMat; }

    pcl::PointCloud<pcl::PointXYZI>::Ptr projectedCloud() const { return m_projectedCloud; }
    pcl::PointCloud<pcl::PointXYZI>::Ptr boundaryPoints() const { return m_boundaryPoints; }
    pcl::PointCloud<pcl::PointXYZI>::Ptr cornerPoints() const { return m_cornerPoints; }
    pcl::PointCloud<pcl::PointXYZI>::Ptr veilPoints() const { return m_veilPoints; }
    pcl::PointCloud<pcl::PointXYZI>::Ptr borderPoints() const { return m_borderPoints; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr voxelPoints() const { return m_voxelPoints; }

    QList<Plane> planes() { return m_planes; }

    QMap<qulonglong, VoxelInfo> voxelInfos() { return m_voxelInfos; }

    /*int OutlierRemovalMeanK() const { return m_outlierRemovalMeanK; }
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

    float projectedRadiusSearch() const { return m_projectedRadiusSearch; }
    void setProjectedRadiusSearch(float _projectedRadiusSearch) { m_projectedRadiusSearch = _projectedRadiusSearch; }

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
    void setPlanePointsRate(float _value) { m_planePointsRate = _value; }*/

private:
    // 输入点云
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_cloud;
    std::vector<int> m_indices;

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

    cv::Mat m_pointsMat;

    // 真正的边界点
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_boundaryPoints;

    // 拐角点
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_cornerPoints;

    // 阴影点
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_veilPoints;

    // 屏幕边缘点
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_borderPoints;

    // voxel cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_voxelPoints;

    // 抽取出的平面
    QList<Plane> m_planes;

    QMap<qulonglong, VoxelInfo> m_voxelInfos;

    /*DOWNSAMPLING_METHOD m_downsamplingMethod;
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
    float m_projectedRadiusSearch;
    float m_veilDistanceThreshold;
    float m_crossPointsRadiusSearch;
    float m_crossPointsClusterTolerance;
    float m_curvatureThreshold;
    int m_minNormalClusters;
    int m_maxNormalClusters;
    float m_planeDistanceThreshold;

    int m_classifyRadius;
    float m_planePointsRate;*/

    PROPERTY(float, Cx)
    PROPERTY(float, Cy)
    PROPERTY(float, Fx)
    PROPERTY(float, Fy)
    PROPERTY(int, Width)
    PROPERTY(int, Height)
    PROPERTY(float, BorderLeft)
    PROPERTY(float, BorderRight)
    PROPERTY(float, BorderTop)
    PROPERTY(float, BorderBottom)
    PROPERTY(float, DepthShift)
    PROPERTY(float, MinDepth)
    PROPERTY(float, MaxDepth)
    PROPERTY(int, CudaNormalKernalRadius)
    PROPERTY(float, CudaNormalKnnRadius)
    PROPERTY(float, CudaBEDistance)
    PROPERTY(float, CudaBEAngleThreshold)
    PROPERTY(int, CudaBEKernalRadius)
    PROPERTY(float, CudaGaussianSigma)
    PROPERTY(int, CudaGaussianKernalRadius)
    PROPERTY(int, CudaClassifyKernalRadius)
    PROPERTY(float, CudaClassifyDistance)
    PROPERTY(int, CudaPeakClusterTolerance)
    PROPERTY(int, CudaMinClusterPeaks)
    PROPERTY(int, CudaMaxClusterPeaks)
    PROPERTY(float, CudaCornerHistSigma)

    PROPERTY(float, VBRGResolution)
    PROPERTY(int, VBRGMinPoints)
};

class BEOctree : public pcl::octree::OctreePointCloudPointVector<pcl::PointXYZ>
{
public:
    /** \brief Constructor.
     *  \param resolution_arg: octree resolution at lowest octree level
     * */
    BEOctree(const double resolution_arg) :
        pcl::octree::OctreePointCloudPointVector<pcl::PointXYZ>(resolution_arg)
    {
    }

    /** \brief Empty class constructor. */
    ~BEOctree()
    {
    }

private:
    QMap<qulonglong, qulonglong> m_relations;
};

#endif // BOUNDARYEXTRACTOR_H
