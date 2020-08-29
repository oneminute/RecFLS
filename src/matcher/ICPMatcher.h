#ifndef ICPMATCHER_H
#define ICPMATCHER_H

#include <QObject>

#include <pcl/common/common.h>
#include <pcl/features/eigen.h>
#include <pcl/search/kdtree.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <cuda/IcpInternal.h>
#include <common/Frame.h>

class ICPMatcher : public QObject
{
    Q_OBJECT
public:
    explicit ICPMatcher(QObject* parent = nullptr);

    Eigen::Matrix4f compute(
        cuda::IcpCache& cache,
        const Eigen::Matrix3f& initRot,
        const Eigen::Vector3f& initTrans,
        float& error);

    Eigen::Matrix4f stepGPU(
        cuda::IcpCache& cache,
        const Eigen::Matrix3f& initRot,
        const Eigen::Vector3f& initTrans,
        float& error);

    Eigen::Matrix4f step(
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloudSrc,
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloudDst,
        pcl::PointCloud<pcl::Normal>::Ptr normalsSrc,
        pcl::PointCloud<pcl::Normal>::Ptr normalsDst,
        pcl::search::KdTree<pcl::PointXYZI>::Ptr tree,
        const Eigen::Matrix3f& initRot,
        const Eigen::Vector3f& initTrans,
        float radius,
        float angleThreshold,
        int& pairsCount,
        float& error);

protected:

private:
    

};

#endif // ICPMATCHER_H