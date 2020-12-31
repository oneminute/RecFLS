#include "LineMatcher.h"

#include <QDebug>
#include <QtMath>

#include <pcl/common/pca.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>

#include <Eigen/SVD>

#include "util/Utils.h"

LineMatcher::LineMatcher(QObject* parent)
    : QObject(parent)
{

}

Eigen::Matrix4f LineMatcher::compute(
    FLFrame& srcFrame
    , FLFrame& dstFrame
    , float& error)
{
    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    QMap<int, int> pairs; 
    QMap<int, float> weights;
    pcl::KdTreeFLANN<LineSegment>::Ptr tree(new pcl::KdTreeFLANN<LineSegment>());

    tree->setInputCloud(dstFrame.lines());
    match(srcFrame.lines(), dstFrame.lines(), tree, pairs, weights);

    for (int i = 0; i < 5; i++)
    {
        Eigen::Matrix4f deltaPose = step(srcFrame.lines(), dstFrame.lines(), pose, error, pairs, weights);
        if (pairs.size() < 3)
        {
            deltaPose = Eigen::Matrix4f::Identity();
            error = 1;
            break;
        }
        pose = deltaPose * pose;
    }

    return srcFrame.pose();
}

void LineMatcher::match(
    pcl::PointCloud<LineSegment>::Ptr srcLines
    , pcl::PointCloud<LineSegment>::Ptr dstLines
    , pcl::KdTreeFLANN<LineSegment>::Ptr tree
    , QMap<int, int>& pairs
    , QMap<int, float>& weights
)
{
    pairs.clear();
    weights.clear();

    QMap<int, float> pairsDists;
    // 通过Kdtree进行匹配直线对的初选。
    for (int i = 0; i < srcLines->points.size(); i++)
    {
        LineSegment lineSrc = srcLines->points[i];

        std::vector<int> indices;
        std::vector<float> dists;
        // 注意下一句，只用Kdtree选取当前直线的一条目标直线。有且只有一条。
        // 在这种情况下，选出的直线可能与源直线误差极大，需要进行筛选。
        if (!tree->nearestKSearch(lineSrc, 1, indices, dists))
            continue;

        std::cout << std::setw(8) << i << " --> " << indices[0] << std::endl;

        // 若源直线集合中的多条直线对应到了同一条目标集合的直线上，
        // 则选取超维向量距离最小的。
        if (pairs.contains(indices[0]))
        {
            if (dists[0] > pairsDists[indices[0]])
            {
                continue;
            }
        }
        pairs[indices[0]] = i;
        pairsDists[indices[0]] = dists[0];
    }
    
    float avgRadians = 0;   // 直线间角度值平均值
    float sqrRadians = 0;   // 直线间角度值平方和
    float avgDist = 0;      // 直线间距离值平均值
    float sqrDist = 0;      // 直线间距离值平方和
    float avgDiff = 0;      // 线段长度差平均值
    float sqrDiff = 0;      // 线段长度差平方和

    // 进行加和
    float avgLength = 0;
    for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end(); i++)
    {
        LineSegment dstLine = dstLines->points[i.key()];
        LineSegment srcLine = srcLines->points[i.value()];
        avgLength += (dstLine.length() + srcLine.length()) / 2;

        float diff = abs(dstLine.length() - srcLine.length());
        avgDiff += diff;
        sqrDiff += diff * diff;

        Eigen::Vector3f dstDir = dstLine.normalizedDir();
        Eigen::Vector3f srcDir = srcLine.normalizedDir();

        Eigen::Vector3f dstPoint = dstLine.middle();
        Eigen::Vector3f srcPoint = srcLine.middle();

        float radians = acos(abs(srcDir.dot(dstDir)));
        avgRadians += radians;
        sqrRadians += radians * radians;

        //float dist = (dstPoint - srcPoint).cross(dstDir).norm();
        float dist = (dstPoint - srcPoint).norm();
        avgDist += dist;
        sqrDist += dist * dist;

        std::cout << std::setw(8) << i.value() << " --> " << i.key() << std::setw(12) << ": radians = "  << radians << ", dist = " << dist << std::endl;
    }
    // 求均值
    avgRadians /= pairs.size();
    avgDist /= pairs.size();
    avgDiff /= pairs.size();
    avgLength /= pairs.size();
    int n = pairs.size();

    // 求角度值与距离值的标准差
    float sdRadians = sqrtf(sqrRadians / n - avgRadians * avgRadians);
    float sdDist = sqrtf(sqrDist / n - avgDist * avgDist);
    float sdDiff = sqrtf(sqrDiff / n - avgDiff * avgDiff);
    //qDebug() << "sdRadians:" << sdRadians << ", sdDist:" << sdDist;

    // 角度值或距离值大于一个标准差的均剔除掉
    QList<int> removedIds;
    for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end(); i++)
    {
        LineSegment dstLine = dstLines->points[i.key()];
        LineSegment srcLine = srcLines->points[i.value()];
        float length = (dstLine.length() + srcLine.length()) / 2;
        float diff = abs(dstLine.length() - srcLine.length());

        Eigen::Vector3f dstDir = dstLine.normalizedDir();
        Eigen::Vector3f srcDir = srcLine.normalizedDir();

        Eigen::Vector3f dstPoint = dstLine.middle();
        Eigen::Vector3f srcPoint = srcLine.middle();

        float radians = acos(abs(srcDir.dot(dstDir)));
        //float dist = (dstPoint - srcPoint).cross(dstDir).norm();
        float dist = (dstPoint - srcPoint).norm();

        if (abs(radians - avgRadians) > sdRadians * 1.f)
        {
            //std::cout << "removed(r): " << i.value() << " --> " << i.key() << ": " << radians << std::endl;
            removedIds.append(i.key());
            continue;
        }
        if (abs(dist - avgDist) > sdDist * 1.f)
        {
            //std::cout << "removed(d): " << i.value() << " --> " << i.key() << ": " << dist << std::endl;
            removedIds.append(i.key());
            continue;
        }
        if (abs(diff - avgDiff) > sdDiff * 1.f)
        {
            //std::cout << "removed(f): " << i.value() << " --> " << i.key() << ": " << diff << std::endl;
            removedIds.append(i.key());
            continue;
        }
        if (length < avgLength * 1.f)
        {
            //std::cout << "removed(l): " << i.value() << " --> " << i.key() << ": " << length << std::endl;
            removedIds.append(i.key());
            continue;
        }
    }

    for (QList<int>::iterator i = removedIds.begin(); i != removedIds.end(); i++)
    {
        pairs.remove(*i);
    }

    float sumLength = 0;
    for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end(); i++)
    {
        LineSegment& dstLine = dstLines->points[i.key()];
        LineSegment& srcLine = srcLines->points[i.value()];

        float degrees = qAcos(dstLine.normalizedDir().dot(srcLine.normalizedDir()));
        if ((M_PI - degrees) < 0.1f)
            srcLine.reverse();
        
        float length = (dstLine.length() + srcLine.length()) / 2;
        weights.insert(i.key(), length);
        sumLength += length;
    }

    for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end(); i++)
    {
        weights[i.key()] /= sumLength;
    }
}

Eigen::Matrix4f LineMatcher::step(
    pcl::PointCloud<LineSegment>::Ptr srcLines
    , pcl::PointCloud<LineSegment>::Ptr dstLines
    , const Eigen::Matrix4f& initPose
    , float& error
    , QMap<int, int>& pairs
    , QMap<int, float>& weights
)
{
    Eigen::Matrix4f out = Eigen::Matrix4f::Identity();

    Eigen::Matrix3f initRot = initPose.topLeftCorner(3, 3);
    Eigen::Vector3f initTrans = initPose.topRightCorner(3, 1);

    Eigen::Matrix4f deltaPose = Eigen::Matrix4f::Identity();
    Eigen::Matrix3f deltaRot = stepRotation(srcLines, dstLines, initRot, pairs, weights);
    Eigen::Vector3f deltaTrans = stepTranslation(srcLines, dstLines, pairs, weights, initRot, initTrans, deltaRot);
    //Eigen::Vector3f deltaTrans = Eigen::Vector3f::Zero();

    deltaPose.topLeftCorner(3, 3) = deltaRot;
    deltaPose.topRightCorner(3, 1) = deltaTrans;

    error = computeError(srcLines, dstLines, pairs, initRot, initTrans, deltaRot, deltaTrans);

    return deltaPose;
}

Eigen::Matrix3f LineMatcher::stepRotation(
    pcl::PointCloud<LineSegment>::Ptr srcLines
    , pcl::PointCloud<LineSegment>::Ptr dstLines
    , const Eigen::Matrix3f& initRot
    , QMap<int, int>& pairs
    , QMap<int, float>& weights
)
{
    Eigen::Vector3f srcAvgDir1(Eigen::Vector3f::Zero());
    Eigen::Vector3f dstAvgDir1(Eigen::Vector3f::Zero());
	Eigen::Vector3f srcAvgDir(Eigen::Vector3f::Zero());
	Eigen::Vector3f dstAvgDir(Eigen::Vector3f::Zero());
    for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end(); i++)
    {
        LineSegment dstLine = dstLines->points[i.key()];
        LineSegment srcLine = srcLines->points[i.value()];

        float degrees = qAcos(dstLine.normalizedDir().dot(srcLine.normalizedDir()));
        std::cout << std::setw(8) << i.value() << " --> " << i.key() << std::setw(12) << ": degrees = " << degrees << ", angles = " << qRadiansToDegrees(degrees) << std::endl;

        dstAvgDir += dstLine.normalizedDir();
		/*dstAvgDir1 = dstLine.normalizedDir()*weights[i.key()];
		dstAvgDir += dstAvgDir1;*/

        srcAvgDir += initRot * srcLine.normalizedDir()  ;
		/*srcAvgDir1 = initRot * srcLine.normalizedDir()*weights[i.key()];
		srcAvgDir += srcAvgDir1;
		*/
    }
    dstAvgDir /= pairs.size();
    srcAvgDir /= pairs.size();
  
	Eigen::Matrix3f cov(Eigen::Matrix3f::Zero());
    for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end(); i++)
    {
        LineSegment dstLine = dstLines->points[i.key()];
        LineSegment srcLine = srcLines->points[i.value()];

        Eigen::Vector3f dstDiff = dstLine.normalizedDir() - dstAvgDir;
        Eigen::Vector3f srcDiff = initRot * srcLine.normalizedDir() - srcAvgDir;
        cov += srcDiff * weights[i.key()] * dstDiff.transpose();
		//cov /= pairs.size();
    }
    
    //Eigen::EigenSolver<Eigen::Matrix3f> eigensolver(cov);
    //Eigen::Matrix3f em = eigensolver.eigenvectors().real();
    //Eigen::Vector3f ev = eigensolver.eigenvalues().real();
    //std::cout << "eigen values:" << ev.transpose() << std::endl;
    //std::cout << "eigen matrix:" << std::endl << (em.colwise() * ev.transpose()) << std::endl;

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(cov, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::Matrix3f V = svd.matrixV();
    Eigen::Matrix3f U = svd.matrixU();
    Eigen::Vector3f sigma = svd.singularValues();
    Eigen::Matrix3f S = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f tmp = V * U.transpose();
    //std::cout << "sigma: " << sigma.transpose() << std::endl;
    float det = tmp.determinant();
    if (det < 0)
        det = -1;
    S.col(2)[2] = det;
    Eigen::Matrix3f R = V * S * U.transpose();
    //Eigen::Matrix3f R = V * U.transpose();
    //qDebug() << "det =" << det;
    //std::cout << "V:" << std::endl << V << std::endl;
    //std::cout << "S:" << std::endl << S << std::endl;
    //std::cout << "U:" << std::endl << U << std::endl;
    //std::cout << "R:" << std::endl << R << std::endl;

    return R;
}

Eigen::Vector3f LineMatcher::stepTranslation(
    pcl::PointCloud<LineSegment>::Ptr srcLines
    , pcl::PointCloud<LineSegment>::Ptr dstLines
    , QMap<int, int>& pairs
    , QMap<int, float>& weights
    , const Eigen::Matrix3f& initRot
    , const Eigen::Vector3f& initTrans
    , const Eigen::Matrix3f& deltaRot
)
{
    Eigen::Vector3f trans(Eigen::Vector3f::Zero());

    for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end(); i++)
    {
        Eigen::Matrix3f iteRot = deltaRot * initRot;
        LineSegment dstLine = dstLines->points[i.key()];
        LineSegment srcLine = srcLines->points[i.value()];
        Eigen::Vector3f dstLineDir = dstLine.normalizedDir();
		Eigen::Vector3f srcLineDir = iteRot * srcLine.normalizedDir();
		
        Eigen::Vector3f diff = dstLine.center() - deltaRot * (initRot * srcLine.center() + initTrans);
        Eigen::Vector3f vertDir = srcLineDir.cross(dstLineDir);
        if (vertDir.norm() <= 0.000001f)
        {
            vertDir = diff.cross(srcLineDir).cross(srcLineDir);
        }
        float dist = diff.dot(vertDir.normalized());
        std::cout << std::setw(8) << i.value() << " --> " << i.key() << std::setw(12) << ": dist = " << dist << ", diff = " << diff.norm() << std::endl;
        Eigen::Vector3f t = vertDir.normalized() * dist;
        //Eigen::Vector3f t = diff;
        trans += t * weights[i.key()];
    }
    //trans /= pairs.count();
    std::cout << "trans: " << trans.transpose() << std::endl;
    
    return trans;
}

float LineMatcher::computeError(
    pcl::PointCloud<LineSegment>::Ptr srcLines
    , pcl::PointCloud<LineSegment>::Ptr dstLines
    , QMap<int, int>& pairs
    , const Eigen::Matrix3f& initRot
    , const Eigen::Vector3f& initTrans
    , const Eigen::Matrix3f& deltaRot
    , const Eigen::Vector3f& deltaTrans
)
{
    float error = 0;
    for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end(); i++)
    {
        LineSegment dstLine = dstLines->points[i.key()];
        LineSegment srcLine = srcLines->points[i.value()];
        Eigen::Vector3f dstLineDir = dstLine.normalizedDir();
        Eigen::Vector3f srcLineDir = deltaRot * initRot * srcLine.normalizedDir();

        Eigen::Vector3f vertLine = srcLineDir.cross(dstLineDir).normalized();
        float distance = (dstLine.middle() - deltaRot * (initRot * srcLine.middle() - initTrans) - deltaTrans).dot(vertLine);
        float degrees = 1 - dstLineDir.dot(srcLineDir);
        error += abs(distance) + abs(degrees);
    }
    error /= pairs.size();

    std::cout << "error: " << error << std::endl;
    return error;
}



