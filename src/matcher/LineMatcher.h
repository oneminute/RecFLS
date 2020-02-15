#ifndef LINEMATCHER_H
#define LINEMATCHER_H

#include <QObject>

#include <pcl/common/common.h>
#include <pcl/features/eigen.h>
#include <pcl/search/kdtree.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <extractor/LineExtractor.h>

class LineMatcher : public QObject
{
    Q_OBJECT
public:
    explicit LineMatcher(QObject* parent = nullptr);

    Eigen::Matrix4f compute(
        QList<LineChain>& chains1,
        pcl::PointCloud<MSL>::Ptr& lines1,
        pcl::PointCloud<LineDescriptor2>::Ptr& desc1,
        QList<LineChain>& chains2,
        pcl::PointCloud<MSL>::Ptr& lines2,
        pcl::PointCloud<LineDescriptor2>::Ptr& desc2
    );

    Eigen::Quaternionf stepRotation(
        float firstDiameter,
        pcl::PointCloud<MSLPoint>::Ptr firstPointCloud,
        pcl::PointCloud<MSL>::Ptr firstLineCloud,
        float secondDiameter,
        pcl::PointCloud<MSLPoint>::Ptr secondPointCloud,
        pcl::PointCloud<MSL>::Ptr secondLineCloud,
        pcl::KdTreeFLANN<MSLPoint>::Ptr tree,
        float& rotationError,
        float& translationError,
        QMap<int, int>& pairs = QMap<int, int>()
    );

    Eigen::Vector3f stepTranslation(
        pcl::PointCloud<MSL>::Ptr firstLineCloud,
        pcl::PointCloud<MSL>::Ptr secondLineCloud,
        pcl::KdTreeFLANN<MSLPoint>::Ptr tree,
        float& translationError,
        QMap<int, int>& pairs = QMap<int, int>()
    );

    QList<LineChain> chains1() const { return m_chains1; }
    QList<LineChain> chains2() const { return m_chains2; }

    QMap<int, int> pairs() const { return m_pairs; }
    QList<int> pairIndices() const { return m_pairIndices; }

protected:

private:
    pcl::PointCloud<LineDescriptor2>::Ptr m_descriptors1;
    pcl::PointCloud<LineDescriptor2>::Ptr m_descriptors2;
    QList<LineChain> m_chains1;
    QList<LineChain> m_chains2;
    QMap<int, int> m_pairs;
    QList<int> m_pairIndices;
};

#endif // LINEMATCHER_H
