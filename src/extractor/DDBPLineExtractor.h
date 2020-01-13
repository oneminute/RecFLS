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

private:
    // 边界点点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_boundaryCloud;

    // 边界点点云索引
    pcl::IndicesPtr m_boundaryIndices;

    // 边界点主方向参数化点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_angleCloud;

    // 边界点主方向点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_dirCloud;

    // 点界点相对于目标线段的参数化点云
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_mappingCloud;

    // 边界点主方向参数化点云每个点的近邻重心
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_centerCloud;

    // 边界点主方向参数化点云每个点的偏心率
    QList<float> m_offsetRate;

    // 边界点主方向参数化点云每个点的近邻密度值
    QList<float> m_density;

    QList<LineSegment> m_lines;

    float m_searchRadius;
    int m_minNeighbourCount;
    float m_searchErrorThreshold;

    Eigen::Vector3f m_maxPoint;
    Eigen::Vector3f m_minPoint;
    float m_boundBoxDiameter;
    
    float m_radiansThreshold;

};

#endif // DDBPLINEEXTRACTOR_H