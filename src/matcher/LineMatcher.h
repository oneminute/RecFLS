#ifndef LINEMATCHER_H
#define LINEMATCHER_H

#include <QObject>
#include <QMap>

#include <pcl/common/common.h>
#include <pcl/features/eigen.h>
#include <pcl/search/kdtree.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <extractor/LineSegment.h>
#include <common/FLFrame.h>

class LineMatcher : public QObject
{
    Q_OBJECT
public:
    explicit LineMatcher(QObject* parent = nullptr);

    Eigen::Matrix4f compute(
        FLFrame& srcFrame
        , FLFrame& dstFrame
        , float& error
        , int& iterations
    );

    Eigen::Matrix4f compute2(
        FLFrame& srcFrame
        , FLFrame& dstFrame
        , float& error
        , int& iterations
    );

    void match(
        pcl::PointCloud<LineSegment>::Ptr srcLines
        , pcl::PointCloud<LineSegment>::Ptr dstLines
        , pcl::KdTreeFLANN<LineSegment>::Ptr tree
        , QMap<int, int>& pairs
        , QMap<int, float>& weights
    );

    Eigen::Matrix4f step(
        pcl::PointCloud<LineSegment>::Ptr srcLines
        , pcl::PointCloud<LineSegment>::Ptr dstLines
        , const Eigen::Matrix4f& initPose
        , float& error
        , QMap<int, int>& pairs
        , QMap<int, float>& weights
    );

    Eigen::Matrix3f stepRotation(
        pcl::PointCloud<LineSegment>::Ptr srcLines
        , pcl::PointCloud<LineSegment>::Ptr dstLines
        , const Eigen::Matrix3f& initRot
        , QMap<int, int>& pairs
        , QMap<int, float>& weights
    );

    Eigen::Vector3f stepTranslation(
        pcl::PointCloud<LineSegment>::Ptr srcLines
        , pcl::PointCloud<LineSegment>::Ptr dstLines
        , QMap<int, int>& pairs
        , QMap<int, float>& weights
        , const Eigen::Matrix3f& initRot
        , const Eigen::Vector3f& initTrans
        , const Eigen::Matrix3f& deltaRot = Eigen::Matrix3f::Identity()
    );

    float computeError(
        pcl::PointCloud<LineSegment>::Ptr srcLines
        , pcl::PointCloud<LineSegment>::Ptr dstLines
        , QMap<int, int>& pairs
        , const Eigen::Matrix3f& initRot
        , const Eigen::Vector3f& initTrans
        , const Eigen::Matrix3f& deltaRot = Eigen::Matrix3f::Identity()
        , const Eigen::Vector3f& deltaTrans = Eigen::Vector3f::Zero()
    );

    Eigen::Vector3f stepTranslation2(
        pcl::PointCloud<LineSegment>::Ptr srcLines,
        pcl::PointCloud<LineSegment>::Ptr dstLines,
        pcl::KdTreeFLANN<LineSegment>::Ptr tree,
        QMap<int, int>& pairs = QMap<int, int>(),
        const Eigen::Matrix3f& initRot = Eigen::Matrix3f::Identity(),
        const Eigen::Vector3f& initTrans = Eigen::Vector3f::Zero(),
        const Eigen::Matrix3f& rot = Eigen::Matrix3f::Identity()
    );

    void extractLineChains(
        pcl::PointCloud<LineSegment>::Ptr srcLines,
        pcl::PointCloud<LineSegment>::Ptr dstLines,
        QMap<int, int>& pairs,
        QMap<int, int>& chains
    );

    QMap<int, int> pairs() const { return m_pairs; }
    QList<int> pairIndices() const { return m_pairIndices; }

protected:

private:
    QMap<int, int> m_pairs;
    QList<int> m_pairIndices;

};

#endif // LINEMATCHER_H
