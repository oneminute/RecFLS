#ifndef LINEEXTRACTOR_H
#define LINEEXTRACTOR_H

#include <QObject>
#include <QList>

#include <pcl/common/common.h>
#include <pcl/features/eigen.h>
#include <pcl/search/kdtree.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "LineSegment.h"
#include "BoundaryExtractor.h"

struct MSLPoint
{
    union
    {
        float props[5];
        struct
        {
            float alpha;
            float beta;
            float x;
            float y;
            float z;
        };
    };
    static int propsSize() { return 5; }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

struct MSL
{
    Eigen::Vector3f dir;
    Eigen::Vector3f point;
    float weight;

    Eigen::Vector3f getEndPoint(float length)
    {
        return point + dir * length;
    }
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class LineExtractor : public QObject
{
    Q_OBJECT
public:
    enum ANGLE_MAPPING_METHOD
    {
        TWO_DIMS = 0,
        THREE_DIMS
    };
    
    explicit LineExtractor(QObject* parent = nullptr);

    QList<LineSegment> compute(const pcl::PointCloud<pcl::PointXYZI>::Ptr& boundaryCloud);

    void extractLinesFromPlanes(const QList<Plane>& planes);

    pcl::PointCloud<pcl::PointXYZI>::Ptr boundaryCloud() const
    {
        return m_boundaryCloud;
    }

    pcl::IndicesPtr boundaryIndices() const
    {
        return m_boundaryIndices;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr angleCloud() const
    {
        return m_angleCloud;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr dirCloud() const
    {
        return m_dirCloud;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr mappingCloud() const
    {
        return m_mappingCloud;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr centerCloud() const
    {
        return m_centerCloud;
    }

    pcl::PointCloud<pcl::PointXYZI>::Ptr linedCloud() const
    {
        return m_linedCloud;
    }

    pcl::PointCloud<MSLPoint>::Ptr mslPointCloud() const
    {
        return m_mslPointCloud;
    }

    pcl::PointCloud<MSL>::Ptr mslCloud() const
    {
        return m_mslCloud;
    }

    QList<float> densityList() const
    {
        return m_density;
    }

    QList<int> angleCloudIndices() const
    {
        return m_angleCloudIndices;
    }

    QMap<int, std::vector<int>> subCloudIndices() const
    {
        return m_subCloudIndices;
    }

    QList<float> errors() const
    {
        return m_errors;
    }

    QList<int> linePointsCount() const
    {
        return m_linePointsCount;
    }

    float boundBoxDiameter() const { return m_boundBoxDiameter; }

    float searchRadius() const { return m_searchRadius; }
    void setSearchRadius(float _searchRadius) { m_searchRadius = _searchRadius; }

    int minNeighbours() const { return m_minNeighbours; }
    void setMinNeighbours(int _minNeighbours) { m_minNeighbours = _minNeighbours; }

    float searchErrorThreshold() const { return m_searchErrorThreshold; }
    void setSearchErrorThreshold(float _searchErrorThreshold) { m_searchErrorThreshold = _searchErrorThreshold; }

    float angleSearchRadius() const { return m_angleSearchRadius; }
    void setAngleSearchRadius(float _angleSearchRadius) { m_angleSearchRadius = _angleSearchRadius; }

    int angleMinNeighbours() const { return m_angleMinNeighbours; }
    void setAngleMinNeighbours(int _angleMinNeighbours) { m_minNeighbours = _angleMinNeighbours; }

    float mappingTolerance() const { return m_mappingTolerance; }
    void setMappingTolerance(float _mappingTolerance) { m_mappingTolerance = _mappingTolerance; }

    float regionGrowingZDistanceThreshold() const { return m_regionGrowingZDistanceThreshold; }
    void setRegionGrowingZDistanceThreshold(float _regionGrowingZDistance) { m_regionGrowingZDistanceThreshold = _regionGrowingZDistance; }

    ANGLE_MAPPING_METHOD angleMappingMethod() const { return m_angleMappingMethod; }
    void setAngleMappingMethod(int _angleMappingMethod) { m_angleMappingMethod = static_cast<ANGLE_MAPPING_METHOD>(_angleMappingMethod); }

    float minLineLength() const { return m_minLineLength; }
    void setMinLineLength(float _minLineLength) { m_minLineLength = _minLineLength; }

    float mslRadiusSearch() const { return m_mslRadiusSearch; }
    void setMslRadiusSearch(float _mslRadiusSearch) { m_mslRadiusSearch = _mslRadiusSearch; }

protected:

private:
    // 边界点点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_boundaryCloud;

    // 边界点点云索引
    pcl::IndicesPtr m_boundaryIndices;

    QList<int> m_angleCloudIndices;

    // 边界点主方向参数化点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_angleCloud;

    // 边界点主方向点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_dirCloud;

    // 点界点相对于目标线段的参数化点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_mappingCloud;

    // 边界点主方向参数化点云每个点的近邻重心
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_centerCloud;

    pcl::PointCloud<pcl::PointXYZI>::Ptr m_linedCloud;

    pcl::PointCloud<MSLPoint>::Ptr m_mslPointCloud;

    pcl::PointCloud<MSL>::Ptr m_mslCloud;

    QMap<int, std::vector<int>> m_subCloudIndices;

    // 边界点主方向参数化点云每个点的偏心率
    QList<float> m_offsetRate;

    // 边界点主方向参数化点云每个点的近邻密度值
    QList<float> m_density;

    QList<LineSegment> m_lines;

    QList<float> m_errors;

    QList<int> m_linePointsCount;

    ANGLE_MAPPING_METHOD m_angleMappingMethod;

    float m_searchRadius;
    int m_minNeighbours;
    float m_searchErrorThreshold;

    float m_angleSearchRadius;
    int m_angleMinNeighbours;

    float m_mappingTolerance;

    float m_regionGrowingZDistanceThreshold;

    float m_minLineLength;

    float m_mslRadiusSearch;

    Eigen::Vector3f m_maxPoint;
    Eigen::Vector3f m_minPoint;
    float m_boundBoxDiameter;

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif // LINEEXTRACTOR_H