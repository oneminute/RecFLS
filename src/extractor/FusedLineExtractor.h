#ifndef FUSEDLINEEXTRACTOR_H
#define FUSEDLINEEXTRACTOR_H

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
#include "common/Frame.h"
#include "cuda/CudaInternal.h"
#include "cuda/FusedLineInternal.h"
#include "cuda/CudaInternal.h"

struct LS3D
{
    pcl::PointXYZ start;
    pcl::PointXYZ end;
    pcl::PointXYZ center;

    float length() {
        Eigen::Vector3f s = start.getArray3fMap();
        Eigen::Vector3f e = end.getArray3fMap();
        return (s - e).norm();
    }
};

class FusedLineExtractor : public QObject
{
    Q_OBJECT
public:
    explicit FusedLineExtractor(QObject* parent = nullptr);
    ~FusedLineExtractor();

    void computeGPU(cuda::FusedLineFrame& frame);
    void compute(Frame& frame, cuda::GpuFrame& frameGpu);

    void init();

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud() { return m_cloud; }
    pcl::PointCloud<pcl::PointXYZI>::Ptr allBoundary() { return m_allBoundary; }
    pcl::PointCloud<pcl::Normal>::Ptr normals() { return m_normals; }
    QMap<int, LS3D> lines() { return m_lines; }
    QMap<int, pcl::PointCloud<pcl::PointXYZI>::Ptr>& groupPoints() { return m_groupPoints; }

private:
    bool m_init;
    cv::Mat m_boundaryMat;
    cv::Mat m_pointsMat;
    pcl::PointCloud<pcl::PointXYZ>::Ptr m_cloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr m_allBoundary;
    pcl::PointCloud<pcl::Normal>::Ptr m_normals;
    QMap<int, LS3D> m_lines;
    QMap<int, pcl::PointCloud<pcl::PointXYZI>::Ptr> m_groupPoints;
};

#endif // FUSEDLINEEXTRACTOR_H
