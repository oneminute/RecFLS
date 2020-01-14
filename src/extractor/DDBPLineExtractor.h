#ifndef DDBPLINEEXTRACTOR_H
#define DDBPLINEEXTRACTOR_H

#include <QObject>
#include <QList>

#include <pcl/common/common.h>
#include <pcl/features/eigen.h>
#include <pcl/search/kdtree.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "LineSegment.h"

class DDBPLineExtractor : public QObject
{
    Q_OBJECT
public:
    explicit DDBPLineExtractor(QObject* parent = nullptr);

    QList<LineSegment> compute(const pcl::PointCloud<pcl::PointXYZI>::Ptr& boundaryCloud);

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

protected:
    bool customRegionGrowing(const pcl::PointXYZI& ptA, const pcl::PointXYZI& ptB, float sqrDistance);

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

    QMap<int, std::vector<int>> m_subCloudIndices;

    // 边界点主方向参数化点云每个点的偏心率
    QList<float> m_offsetRate;

    // 边界点主方向参数化点云每个点的近邻密度值
    QList<float> m_density;

    QList<LineSegment> m_lines;

    QList<float> m_errors;

    QList<int> m_linePointsCount;

    float m_searchRadius;
    int m_minNeighbours;
    float m_searchErrorThreshold;

    float m_angleSearchRadius;
    int m_angleMinNeighbours;

    float m_mappingTolerance;

    Eigen::Vector3f m_maxPoint;
    Eigen::Vector3f m_minPoint;
    float m_boundBoxDiameter;

};

#endif // DDBPLINEEXTRACTOR_H