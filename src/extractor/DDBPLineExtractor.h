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
    // �߽�����
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_boundaryCloud;

    // �߽���������
    pcl::IndicesPtr m_boundaryIndices;

    // �߽�����������������
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_angleCloud;

    // �߽�����������
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_dirCloud;

    // ���������Ŀ���߶εĲ���������
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_mappingCloud;

    // �߽�����������������ÿ����Ľ�������
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_centerCloud;

    // �߽�����������������ÿ�����ƫ����
    QList<float> m_offsetRate;

    // �߽�����������������ÿ����Ľ����ܶ�ֵ
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