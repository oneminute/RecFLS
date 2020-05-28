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

//Eigen::Matrix4f LineMatcher::compute(
//    FLFrame& srcFrame
//    , FLFrame& dstFrame
//    , float& error)
//{
//    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
//    pcl::KdTreeFLANN<LineSegment>::Ptr tree(new pcl::KdTreeFLANN<LineSegment>());
//    tree->setInputCloud(dstFrame.lines());
//
//    QMap<int, int> pairs;
//
//    for (int i = 0; i < 5; i++)
//    {
//        Eigen::Matrix4f stepPose = step(srcFrame.lines(), dstFrame.lines(), tree, error, pairs);
//        if (pairs.size() < 3)
//        {
//            stepPose = Eigen::Matrix4f::Identity();
//            error = 1;
//            break;
//        }
//        srcFrame.transform(stepPose);
//    }
//
//    return srcFrame.pose();
//}

void LineMatcher::match(
    pcl::PointCloud<LineSegment>::Ptr srcLines
    , pcl::PointCloud<LineSegment>::Ptr dstLines
    , pcl::KdTreeFLANN<LineSegment>::Ptr tree
    , const Eigen::Matrix3f& rot
    , const Eigen::Vector3f& trans
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
        Eigen::Vector3f srcDir = rot * srcLine.normalizedDir();

        Eigen::Vector3f dstPoint = dstLine.middle();
        Eigen::Vector3f srcPoint = rot * srcLine.middle() + trans;

        float radians = acos(abs(srcDir.dot(dstDir)));
        avgRadians += radians;
        sqrRadians += radians * radians;

        //float dist = (dstPoint - srcPoint).cross(dstDir).norm();
        float dist = (dstPoint - srcPoint).norm();
        avgDist += dist;
        sqrDist += dist * dist;

        //qDebug().noquote() << i.value() << "-->" << i.key() << ":" << radians << dist;
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
        Eigen::Vector3f srcDir = rot * srcLine.normalizedDir();

        Eigen::Vector3f dstPoint = dstLine.middle();
        Eigen::Vector3f srcPoint = rot * srcLine.middle() + trans;

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
        LineSegment dstLine = dstLines->points[i.key()];
        LineSegment srcLine = srcLines->points[i.value()];
        
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
    , pcl::KdTreeFLANN<LineSegment>::Ptr tree
    , const Eigen::Matrix4f& initPose
    , float& error
    , QMap<int, int>& pairs
    , QMap<int, float>& weights
)
{
    Eigen::Matrix4f out = Eigen::Matrix4f::Identity();

    Eigen::Matrix3f initRot = initPose.topLeftCorner(3, 3);
    Eigen::Vector3f initTrans = initPose.topRightCorner(3, 1);
    match(srcLines, dstLines, tree, initRot, initTrans, pairs, weights);
    if (pairs.size() < 3)
    {
        return out;
    }

    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    Eigen::Matrix3f rot = stepRotation(srcLines, dstLines, tree, initRot, pairs, weights);
    Eigen::Vector3f trans = stepTranslation(srcLines, dstLines, tree, pairs, weights, initRot, initTrans, rot);

    pose.topLeftCorner(3, 3) = rot;
    pose.topRightCorner(3, 1) = trans;

    error = computeError(srcLines, dstLines, tree, pairs, initRot, initTrans, rot, trans);

    return pose;
}

Eigen::Matrix3f LineMatcher::stepRotation(
    pcl::PointCloud<LineSegment>::Ptr srcLines
    , pcl::PointCloud<LineSegment>::Ptr dstLines
    , pcl::KdTreeFLANN<LineSegment>::Ptr tree
    , const Eigen::Matrix3f& initRot
    , QMap<int, int>& pairs
    , QMap<int, float>& weights
)
{
    Eigen::Vector3f srcAvgDir(Eigen::Vector3f::Zero());
    Eigen::Vector3f dstAvgDir(Eigen::Vector3f::Zero());
    for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end(); i++)
    {
        LineSegment dstLine = dstLines->points[i.key()];
        LineSegment srcLine = srcLines->points[i.value()];
        dstAvgDir += dstLine.normalizedDir() * weights[i.key()];
        srcAvgDir += initRot * srcLine.normalizedDir() * weights[i.key()];
    }

    Eigen::Matrix3f cov(Eigen::Matrix3f::Zero());
    for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end(); i++)
    {
        LineSegment dstLine = dstLines->points[i.key()];
        LineSegment srcLine = srcLines->points[i.value()];

        Eigen::Vector3f dstDiff = dstLine.normalizedDir() - dstAvgDir;
        Eigen::Vector3f srcDiff = initRot * srcLine.normalizedDir() - srcAvgDir;
        cov += srcDiff * dstDiff.transpose() * weights[i.key()];
    }

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
    , pcl::KdTreeFLANN<LineSegment>::Ptr tree
    , QMap<int, int>& pairs
    , QMap<int, float>& weights
    , const Eigen::Matrix3f& initRot
    , const Eigen::Vector3f& initTrans
    , const Eigen::Matrix3f& rot
)
{
    Eigen::Vector3f trans(Eigen::Vector3f::Zero());

    Eigen::Matrix3f A(Eigen::Matrix3f::Zero());
    Eigen::Vector3f b(Eigen::Vector3f::Zero());
    Eigen::Vector3f avgV(Eigen::Vector3f::Zero());
    for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end(); i++)
    {
        LineSegment dstLine = dstLines->points[i.key()];
        LineSegment srcLine = srcLines->points[i.value()];
        Eigen::Vector3f dstLineDir = dstLine.normalizedDir();
        Eigen::Vector3f srcLineDir = rot * initRot * srcLine.normalizedDir();

        //Eigen::Vector3f lineDir = (dstLineDir + initRot * srcLineDir) / 2;
        Eigen::Vector3f lineDir = dstLineDir;

        float a2 = lineDir.x() * lineDir.x();
        float b2 = lineDir.y() * lineDir.y();
        float c2 = lineDir.z() * lineDir.z();
        float ab = lineDir.x() * lineDir.y();
        float ac = lineDir.x() * lineDir.z();
        float bc = lineDir.y() * lineDir.z();
        Eigen::Vector3f v = dstLine.middle() - rot * (initRot * srcLine.middle() + initTrans);
        avgV += v * weights[i.key()];
        //float xv = v.x();
        //float yv = v.y();
        //float zv = v.z();

        Eigen::Matrix3f deltaA;
        deltaA << b2 + c2,     -ab,     -ac,
                      -ab, a2 + c2,     -bc,
                      -ac,     -bc, a2 + b2;

        //A.row(0)[0] += b2 + c2; A.row(0)[1] +=     -ab; A.row(0)[2] +=     -ac;
        //A.row(1)[0] +=     -ab; A.row(1)[1] += a2 + c2; A.row(1)[2] +=     -bc;
        //A.row(2)[0] +=     -ac; A.row(2)[1] +=     -bc; A.row(2)[2] += a2 + b2;
        A += deltaA * weights[i.key()];

        //b.x() += (b2 + c2) * xv         -ab * yv         -ac * zv;
        //b.y() +=       -ab * xv + (a2 + c2) * yv         -bc * zv;
        //b.z() +=       -ac * xv         -bc * yv + (a2 + b2) * zv;
        b += deltaA * v * weights[i.key()];
    }
    trans = A.colPivHouseholderQr().solve(b);
    //trans = A.inverse() * b;
    //avgV /= pairs.size();
    //std::cout << " avgV: " << avgV.transpose() << std::endl;
    //std::cout << "trans: " << trans.transpose() << std::endl;
    
    return trans;
}

float LineMatcher::computeError(
    pcl::PointCloud<LineSegment>::Ptr srcLines
    , pcl::PointCloud<LineSegment>::Ptr dstLines
    , pcl::KdTreeFLANN<LineSegment>::Ptr tree
    , QMap<int, int>& pairs
    , const Eigen::Matrix3f& initRot
    , const Eigen::Vector3f& initTrans
    , const Eigen::Matrix3f& rot
    , const Eigen::Vector3f& trans
)
{
    float error = 0;
    for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end(); i++)
    {
        LineSegment dstLine = dstLines->points[i.key()];
        LineSegment srcLine = srcLines->points[i.value()];
        Eigen::Vector3f dstLineDir = dstLine.normalizedDir();
        Eigen::Vector3f srcLineDir = rot * (initRot * srcLine.normalizedDir() + initTrans);

        Eigen::Vector3f vertLine = srcLineDir.cross(dstLineDir).normalized();
        float distance = (dstLine.middle() - rot * (initRot * srcLine.middle() - initTrans) - trans).dot(vertLine);
        error += abs(distance);
    }
    error /= pairs.size();

    //std::cout << "error: " << error << std::endl;
    return error;
}

Eigen::Vector3f LineMatcher::stepTranslation2(pcl::PointCloud<LineSegment>::Ptr srcLines, pcl::PointCloud<LineSegment>::Ptr dstLines, pcl::KdTreeFLANN<LineSegment>::Ptr tree, QMap<int, int>& pairs, const Eigen::Matrix3f& initRot, const Eigen::Vector3f& initTrans, const Eigen::Matrix3f& rot)
{
    Eigen::Vector3f trans(Eigen::Vector3f::Zero());
    QMap<int, int> chains;
    extractLineChains(srcLines, dstLines, pairs, chains);
    for (QMap<int, int>::iterator i = chains.begin(); i != chains.end(); i++)
    {
        LineSegment dstLine1 = dstLines->points[i.key()];
        LineSegment dstLine2 = dstLines->points[i.value()];
        LineSegment srcLine1 = srcLines->points[pairs[i.key()]];
        LineSegment srcLine2 = srcLines->points[pairs[i.value()]];

        Eigen::Vector3f dstLineDir1 = dstLine1.direction().normalized();
        Eigen::Vector3f dstLineDir2 = dstLine2.direction().normalized();
        Eigen::Vector3f srcLineDir1 = srcLine1.direction().normalized();
        Eigen::Vector3f srcLineDir2 = srcLine2.direction().normalized();

        Eigen::Vector3f srcPoint1 = rot * (initRot * srcLine1.middle() + initTrans);
        Eigen::Vector3f dstPoint1 = dstLine1.middle();
        Eigen::Vector3f v1 = srcLineDir1.cross((dstPoint1 - srcPoint1).cross(srcLineDir1)).normalized();
        v1 = v1 * (dstPoint1 - srcPoint1).dot(v1);

        Eigen::Vector3f srcPoint2 = rot * (initRot * srcLine2.middle() + initTrans) + v1;
        Eigen::Vector3f dstPoint2 = dstLine2.middle();
        Eigen::Vector3f v2_ = srcLineDir1.cross((dstPoint2 - srcPoint2).cross(srcLineDir2)).normalized();
        float cosValue = v2_.dot(dstLineDir2);
        float v2_length = (dstPoint2 - srcPoint2).dot(v2_);
        v2_ = v2_ * v2_length;
        float v2Length = v2_length / cosValue;
        Eigen::Vector3f v2 = dstLineDir2 * v2Length;
        
        trans += v1 + v2;
    }
    trans /= chains.size();

    //std::cout << "trans: " << trans.transpose() << std::endl;

    float error = 0;
    for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end(); i++)
    {
        LineSegment dstLine = dstLines->points[i.key()];
        LineSegment srcLine = srcLines->points[i.value()];
        Eigen::Vector3f dstLineDir = dstLine.direction().normalized();
        Eigen::Vector3f srcLineDir = initRot * srcLine.direction().normalized();

        Eigen::Vector3f vertLine = srcLineDir.cross(dstLineDir).normalized();
        float distance = (dstLine.middle() - (initRot * srcLine.middle() + initTrans + trans)).dot(vertLine);
        //std::cout << i.value() << " --> " << i.key() << ": " << distance << std::endl;
        error += distance;
    }
    error /= pairs.size();

    //std::cout << "error: " << error << std::endl;
    return trans;
}

void LineMatcher::extractLineChains(pcl::PointCloud<LineSegment>::Ptr srcLines, pcl::PointCloud<LineSegment>::Ptr dstLines, QMap<int, int>& pairs, QMap<int, int>& chains)
{
    chains.clear();
    for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end(); i++)
    {
        LineSegment dstLine1 = dstLines->points[i.key()];
        for (QMap<int, int>::iterator j = i; j != pairs.end(); j++)
        {
            if (j == i)
            {
                continue;
            }

            LineSegment dstLine2 = dstLines->points[j.key()];

            float radians = acosf(abs(dstLine1.direction().normalized().dot(dstLine2.direction().normalized())));
            if (radians >= M_PI_4)
            {
                chains.insert(dstLine1.index(), dstLine2.index());
                //std::cout << "[" << dstLine1.index() << ", " << dstLine2.index() << "]: " << qRadiansToDegrees(radians) << std::endl;
            }
        }
    }
}

