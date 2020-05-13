#include "ICPMatcher.h"

#include <QObject>
#include <QFile>

#include "cuda/IcpInternal.h"
#include "cuda/cuda.hpp"
#include "common/Parameters.h"

ICPMatcher::ICPMatcher(QObject* parent)
    : QObject(parent)
    , PROPERTY_INIT(MaxIterations, 10)
{
    
}

Eigen::Matrix4f ICPMatcher::compute(cuda::IcpCache& cache, const Eigen::Matrix3f& initRot, const Eigen::Vector3f& initTrans, float& error)
{
    int maxIterations = Settings::ICPMatcher_MaxIterations.intValue();
    Eigen::Matrix4f pose(Eigen::Matrix4f::Identity());
    for (int i = 0; i < maxIterations; i++)
    {
        Eigen::Matrix4f poseDelta = stepGPU(cache, initRot, initTrans, error);
        pose = poseDelta * pose;

    }
    return pose;
}

Eigen::Matrix4f ICPMatcher::stepGPU(
    cuda::IcpCache& cache,
    const Eigen::Matrix3f& initRot,
    const Eigen::Vector3f& initTrans,
    float& error)
{
    Eigen::Vector3f avgSrc;
    Eigen::Vector3f avgDst;
    Eigen::Matrix3f covMatrix;
    float3 gpuAvgSrc, gpuAvgDst;
    Mat33 gpuCovMatrix;
    int pairsCount;
    cuda::icp(cache, toMat33(initRot), toFloat3(initTrans), gpuCovMatrix, gpuAvgSrc, gpuAvgDst, pairsCount);
    avgSrc = toVector3f(gpuAvgSrc);
    avgDst = toVector3f(gpuAvgDst);
    covMatrix = toMatrix3f(gpuCovMatrix);

    std::cout << "covMatrix:" << sizeof(Eigen::Matrix3f) << sizeof(covMatrix) << std::endl << covMatrix << std::endl;

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(covMatrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix3f V = svd.matrixV();
    Eigen::Matrix3f U = svd.matrixU();
    Eigen::Vector3f sigma = svd.singularValues();
    Eigen::Matrix3f S = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f tmp = V * U.transpose();
    std::cout << "sigma: " << sigma.transpose() << std::endl;
    float det = tmp.determinant();
    if (det < 0)
        det = -1;
    S.col(2)[2] = det;
    Eigen::Matrix3f R = V * S * U.transpose();
    qDebug() << "det =" << det;
    std::cout << "V:" << std::endl << V << std::endl;
    std::cout << "S:" << std::endl << S << std::endl;
    std::cout << "U:" << std::endl << U << std::endl;

    Eigen::Matrix4f pose(Eigen::Matrix4f::Identity());
    pose.topLeftCorner(3, 3) = R;

    Eigen::Vector3f T = avgDst - R * avgSrc;
    pose.topRightCorner(3, 1) = T;

    std::cout << "pose:" << std::endl << pose << std::endl;

    cuda::calculateErrors(cache, toMat33(R), toFloat3(T), pairsCount, error);
    return pose;
}

Eigen::Matrix4f ICPMatcher::step(
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
    float& error)
{
    Eigen::Vector3f avgSrc = Eigen::Vector3f::Zero();
    Eigen::Vector3f avgDst = Eigen::Vector3f::Zero();
    pairsCount = 0;
    Eigen::Matrix3f w = Eigen::Matrix3f::Zero();
    
    std::cout << "out result" << std::endl;
    QFile file("cpu_pairs.txt");
    file.open(QIODevice::WriteOnly | QFile::Truncate | QFile::Text);
    QTextStream out(&file);
    QMap<int, int> pairs;
    for (int i = 0; i < cloudSrc->points.size(); i++)
    {
        pcl::PointXYZI ptSrc = cloudSrc->points[i];
        pcl::Normal nmSrc = normalsSrc->points[i];
        Eigen::Vector3f srcPt = initRot * ptSrc.getVector3fMap() + initTrans;
        Eigen::Vector3f srcNm = initRot * nmSrc.getNormalVector3fMap();

        pcl::PointXYZI tmp;
        tmp.getVector3fMap() = srcPt;
        std::vector<int> indices;
        std::vector<float> dists;
        if (!tree->radiusSearch(tmp, radius, indices, dists))
            continue;

        Eigen::Vector3f dstPt;
        Eigen::Vector3f dstNm;
        bool found = false;
        float maxCos = 0;
        int dstIndex = -1;
        pcl::PointXYZI ptDst;
        for (int ni = 0; ni < indices.size(); ni++)
        {
            pcl::PointXYZI ptNeighbour = cloudDst->points[indices[ni]];
            pcl::Normal nmNeighbour = normalsDst->points[indices[ni]];

            float cos = srcNm.dot(nmNeighbour.getNormalVector3fMap());
            if (cos >= angleThreshold)
            {
                if (cos > maxCos)
                {
                    ptDst = ptNeighbour;
                    dstPt = ptNeighbour.getArray3fMap();
                    dstNm = nmNeighbour.getNormalVector3fMap();
                    maxCos = cos;
                    dstIndex = indices[ni];
                    found = true;
                }
            }
        }
        if (!found)
            continue;

        avgSrc += initRot * srcPt + initTrans;
        avgDst += dstPt;

        pairs.insert(ptSrc.intensity, ptDst.intensity);
        int x1, y1, x2, y2;
        x1 = static_cast<int>(ptSrc.intensity) % 640;
        y1 = static_cast<int>(ptSrc.intensity) / 640;
        x2 = static_cast<int>(ptDst.intensity) % 640;
        y2 = static_cast<int>(ptDst.intensity) / 640;
        out << "[" << x1 << ", " << y1 << "] -- [" << x2 << ", " << y2 << "]\n";
    }
    file.close();

    avgSrc /= pairs.size();
    avgDst /= pairs.size();
    std::cout << "avgSrc = " << avgSrc.transpose() << std::endl;
    std::cout << "avgDst = " << avgDst.transpose() << std::endl;

    for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end(); i++)
    {
        pcl::PointXYZI ptSrc = cloudSrc->points[i.key()];
        Eigen::Vector3f srcPt = initRot * ptSrc.getVector3fMap() + initTrans;

        pcl::PointXYZI ptDst = cloudDst->points[i.value()];
        Eigen::Vector3f dstPt = ptDst.getVector3fMap();

        Eigen::Matrix3f m = (srcPt - avgSrc) * (dstPt - avgDst).transpose();
        w += m;
    }

    pairsCount = pairs.size();
    w /= pairs.size();
    std::cout << "w = " << std::endl;
    std::cout << w << std::endl;
    Eigen::JacobiSVD<Eigen::MatrixXf> svd(w, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix3f V = svd.matrixV();
    Eigen::Matrix3f U = svd.matrixU();
    Eigen::Vector3f sigma = svd.singularValues();
    Eigen::Matrix3f S = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f tmp = V * U.transpose();
    std::cout << "sigma: " << sigma.transpose() << std::endl;
    float det = tmp.determinant();
    if (det < 0)
        det = -1;
    S.col(2)[2] = det;
    Eigen::Matrix3f R = V * S * U.transpose();
    qDebug() << "det =" << det;
    std::cout << "V:" << std::endl << V << std::endl;
    std::cout << "S:" << std::endl << S << std::endl;
    std::cout << "U:" << std::endl << U << std::endl;

    Eigen::Matrix4f pose(Eigen::Matrix4f::Identity());
    pose.topLeftCorner(3, 3) = R;

    Eigen::Vector3f T = avgDst - R * avgSrc;
    pose.topRightCorner(3, 1) = T;

    std::cout << "pose:" << std::endl << pose << std::endl;
    return pose;
}

