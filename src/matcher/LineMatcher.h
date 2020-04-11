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

    /*Eigen::Matrix4f compute(
        QList<LineChain>& chains1,
        pcl::PointCloud<Line>::Ptr& lines1,
        pcl::PointCloud<LineDescriptor3>::Ptr& desc1,
        QList<LineChain>& chains2,
        pcl::PointCloud<Line>::Ptr& lines2,
        pcl::PointCloud<LineDescriptor3>::Ptr& desc2
    );*/

    Eigen::Matrix3f stepRotation(
        pcl::PointCloud<Line>::Ptr firstLineCloud,
        pcl::PointCloud<Line>::Ptr secondLineCloud,
        pcl::KdTreeFLANN<Line>::Ptr tree,
        float& rotationError,
        float& translationError,
        QMap<int, int>& pairs = QMap<int, int>()
    );

    Eigen::Vector3f stepTranslation(
        pcl::PointCloud<Line>::Ptr firstLineCloud,
        pcl::PointCloud<Line>::Ptr secondLineCloud,
        pcl::KdTreeFLANN<Line>::Ptr tree,
        float& translationError,
        QMap<int, int>& pairs = QMap<int, int>()
    );

    QList<LineChain> chains1() const { return m_chains1; }
    QList<LineChain> chains2() const { return m_chains2; }

    QMap<int, int> pairs() const { return m_pairs; }
    QList<int> pairIndices() const { return m_pairIndices; }

protected:

private:
    pcl::PointCloud<LineDescriptor>::Ptr m_descriptors1;
    pcl::PointCloud<LineDescriptor>::Ptr m_descriptors2;
    QList<LineChain> m_chains1;
    QList<LineChain> m_chains2;
    QMap<int, int> m_pairs;
    QList<int> m_pairIndices;
};

#endif // LINEMATCHER_H
