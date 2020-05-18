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

Eigen::Matrix4f LineMatcher::compute(pcl::PointCloud<LineSegment>::Ptr srcLines, pcl::PointCloud<LineSegment>::Ptr dstLines
    , const Eigen::Matrix3f& initRot
    , const Eigen::Vector3f& initTrans
    , float& rotationError, float& translationError)
{
    Eigen::Matrix4f out = Eigen::Matrix4f::Identity();
    pcl::KdTreeFLANN<LineSegment>::Ptr tree(new pcl::KdTreeFLANN<LineSegment>());
    tree->setInputCloud(dstLines);

    QMap<int, int> pairs;
    QMap<int, float> pairsDists;
    for (int i = 0; i < srcLines->points.size(); i++)
    {
        LineSegment lineSrc = srcLines->points[i];

        std::vector<int> indices;
        std::vector<float> dists;
        if (!tree->nearestKSearch(lineSrc, 1, indices, dists))
            continue;

        //LineSegment lineDst = dstLines->points[indices[0]];
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

    /*for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end(); i++)
    {
        qDebug().noquote() << i.value() << "-->" << i.key();
    }*/
    Eigen::Matrix3f rot = initRot;
    Eigen::Vector3f trans = initTrans;
    for (int i = 0; i < 20; i++)
    {
        out = step(srcLines, dstLines, tree, rot, trans, rotationError, translationError, pairs);
        rot = out.topLeftCorner(3, 3);
        trans = out.topRightCorner(3, 1);
    }

    return out;
}

void LineMatcher::match(
    pcl::PointCloud<LineSegment>::Ptr srcLines, 
    pcl::PointCloud<LineSegment>::Ptr dstLines, 
    pcl::KdTreeFLANN<LineSegment>::Ptr tree, 
    const Eigen::Matrix3f& initRot,
    const Eigen::Vector3f& initTrans,
    QMap<int, int>& pairs)
{
    pairs.clear();
    QMap<int, float> pairsDists;
    // 通过Kdtree进行匹配直线对的初选。
    for (int i = 0; i < srcLines->points.size(); i++)
    {
        LineSegment lineSrc = srcLines->points[i];
        lineSrc.generateDescriptor(initRot, initTrans);
        //lineSrc.generateDescriptor();

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

    // 进行加和
    for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end(); i++)
    {
        LineSegment dstLine = dstLines->points[i.key()];
        LineSegment srcLine = srcLines->points[i.value()];

        Eigen::Vector3f dstDir = dstLine.direction().normalized();
        Eigen::Vector3f srcDir = initRot * srcLine.direction().normalized();

        Eigen::Vector3f dstPoint = dstLine.middle();
        Eigen::Vector3f srcPoint = initRot * srcLine.middle() + initTrans;

        float radians = acos(abs(srcDir.dot(dstDir)));
        avgRadians += radians;
        sqrRadians += radians * radians;

        float dist = (dstPoint - srcPoint).cross(dstDir).norm();
        avgDist += dist;
        sqrDist += dist * dist;

        //qDebug().noquote() << i.value() << "-->" << i.key();
    }
    // 求均值
    avgRadians /= pairs.size();
    avgDist /= pairs.size();
    int n = pairs.size();

    // 求角度值与距离值的标准差
    float sdRadians = sqrtf(sqrRadians / n - avgRadians * avgRadians);
    float sdDist = sqrtf(sqrDist / n - avgDist * avgDist);

    // 角度值或距离值小于一个标准差的均剔除掉
    QList<int> removedIds;
    for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end(); i++)
    {
        LineSegment dstLine = dstLines->points[i.key()];
        LineSegment srcLine = srcLines->points[i.value()];

        Eigen::Vector3f dstDir = dstLine.direction().normalized();
        Eigen::Vector3f srcDir = initRot * srcLine.direction().normalized();

        Eigen::Vector3f dstPoint = dstLine.middle();
        Eigen::Vector3f srcPoint = initRot * srcLine.middle() + initTrans;

        float radians = acos(abs(srcDir.dot(dstDir)));
        float dist = (dstPoint - srcPoint).cross(dstDir).norm();

        if (radians > sdRadians)
        {
            std::cout << "removed(r): " << i.value() << " --> " << i.key() << std::endl;
            removedIds.append(i.key());
            continue;
        }
        if (dist > sdDist)
        {
            std::cout << "removed(l): " << i.value() << " --> " << i.key() << std::endl;
            removedIds.append(i.key());
            continue;
        }

        qDebug().noquote() << i.value() << "-->" << i.key();
    }

    for (QList<int>::iterator i = removedIds.begin(); i != removedIds.end(); i++)
    {
        pairs.remove(*i);
    }
}

Eigen::Matrix4f LineMatcher::step(
    pcl::PointCloud<LineSegment>::Ptr srcLines
    , pcl::PointCloud<LineSegment>::Ptr dstLines
    , pcl::KdTreeFLANN<LineSegment>::Ptr tree
    , const Eigen::Matrix3f& initRot
    , const Eigen::Vector3f& initTrans
    , float& rotationError
    , float& translationError
    , QMap<int, int>& pairs)
{
    Eigen::Matrix4f out = Eigen::Matrix4f::Identity();

    match(srcLines, dstLines, tree, initRot, initTrans, pairs);

    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    Eigen::Matrix3f rot = stepRotation(srcLines, dstLines, tree, pairs, initRot);

    Eigen::Vector3f trans = stepTranslation(srcLines, dstLines, tree, pairs, initRot, initTrans, rot);

    pose.topLeftCorner(3, 3) = rot;
    pose.topRightCorner(3, 1) = trans;

    /*for (int i = 0; i < srcLines->points.size(); i++)
    {
        srcLines->points[i].generateDescriptor(0, 3, rot, trans);
    }*/

    return pose;
}

Eigen::Matrix3f LineMatcher::stepRotation(
    pcl::PointCloud<LineSegment>::Ptr srcLines,
    pcl::PointCloud<LineSegment>::Ptr dstLines,
    pcl::KdTreeFLANN<LineSegment>::Ptr tree,
    QMap<int, int>& pairs,
    const Eigen::Matrix3f& initRot)
{
    Eigen::Vector3f srcAvgDir(Eigen::Vector3f::Zero());
    Eigen::Vector3f dstAvgDir(Eigen::Vector3f::Zero());
    for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end(); i++)
    {
        LineSegment dstLine = dstLines->points[i.key()];
        LineSegment srcLine = srcLines->points[i.value()];
        dstAvgDir += dstLine.direction().normalized();
        srcAvgDir += initRot * srcLine.direction().normalized();
    }

    Eigen::Matrix3f cov(Eigen::Matrix3f::Zero());
    for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end(); i++)
    {
        LineSegment dstLine = dstLines->points[i.key()];
        LineSegment srcLine = srcLines->points[i.value()];

        Eigen::Vector3f dstDiff = dstLine.direction().normalized() - dstAvgDir;
        Eigen::Vector3f srcDiff = initRot * srcLine.direction().normalized() - srcAvgDir;
        cov += srcDiff * dstDiff.transpose();
    }

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(cov, Eigen::ComputeThinU | Eigen::ComputeThinV);
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
    //Eigen::Matrix3f R = V * U.transpose();
    //qDebug() << "det =" << det;
    std::cout << "V:" << std::endl << V << std::endl;
    std::cout << "S:" << std::endl << S << std::endl;
    std::cout << "U:" << std::endl << U << std::endl;
    std::cout << "R:" << std::endl << R << std::endl;

    return R;
}

Eigen::Vector3f LineMatcher::stepTranslation(
    pcl::PointCloud<LineSegment>::Ptr srcLines,
    pcl::PointCloud<LineSegment>::Ptr dstLines,
    pcl::KdTreeFLANN<LineSegment>::Ptr tree,
    QMap<int, int>& pairs,
    const Eigen::Matrix3f& initRot,
    const Eigen::Vector3f& initTrans,
    const Eigen::Matrix3f& rot
    )
{
    Eigen::Vector3f trans(Eigen::Vector3f::Zero());

    Eigen::Matrix3f A(Eigen::Matrix3f::Zero());
    Eigen::Vector3f b(Eigen::Vector3f::Zero());
    for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end(); i++)
    {
        LineSegment dstLine = dstLines->points[i.key()];
        LineSegment srcLine = srcLines->points[i.value()];
        Eigen::Vector3f dstLineDir = dstLine.direction().normalized();
        Eigen::Vector3f srcLineDir = rot * initRot * srcLine.direction().normalized();

        //Eigen::Vector3f lineDir = (dstLineDir + initRot * srcLineDir) / 2;
        Eigen::Vector3f lineDir = dstLineDir;

        float a2 = lineDir.x() * lineDir.x();
        float b2 = lineDir.y() * lineDir.y();
        float c2 = lineDir.z() * lineDir.z();
        float ab = lineDir.x() * lineDir.y();
        float ac = lineDir.x() * lineDir.z();
        float bc = lineDir.y() * lineDir.z();
        Eigen::Vector3f v = dstLine.middle() - rot * (initRot * srcLine.middle() + initTrans);
        float xv = v.x();
        float yv = v.y();
        float zv = v.z();

        A.row(0)[0] += b2 + c2; A.row(0)[1] +=     -ab; A.row(0)[2] +=     -ac;
        A.row(1)[0] +=     -ab; A.row(1)[1] += a2 + c2; A.row(1)[2] +=     -bc;
        A.row(2)[0] +=     -ac; A.row(2)[1] +=     -bc; A.row(2)[2] += a2 + b2;

        b.x() += (b2 + c2) * xv         -ab * yv         -ac * zv;
        b.y() +=       -ab * xv + (a2 + c2) * yv         -bc * zv;
        b.z() +=       -ac * xv         -bc * yv + (a2 + b2) * zv;
    }
    //trans = A.colPivHouseholderQr().solve(b);
    trans = A.inverse() * b;
    std::cout << "trans: " << trans.transpose() << std::endl;

    float error = 0;
    for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end(); i++)
    {
        LineSegment dstLine = dstLines->points[i.key()];
        LineSegment srcLine = srcLines->points[i.value()];
        Eigen::Vector3f dstLineDir = dstLine.direction().normalized();
        Eigen::Vector3f srcLineDir = rot * initRot * srcLine.direction().normalized();

        Eigen::Vector3f vertLine = srcLineDir.cross(dstLineDir).normalized();
        float distance = (dstLine.middle() - rot * (initRot * srcLine.middle() + initTrans) - trans).dot(vertLine);
        error += distance;
    }
    error /= pairs.size();

    std::cout << "error: " << error << std::endl;

    return trans;
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

    std::cout << "trans: " << trans.transpose() << std::endl;

    float error = 0;
    for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end(); i++)
    {
        LineSegment dstLine = dstLines->points[i.key()];
        LineSegment srcLine = srcLines->points[i.value()];
        Eigen::Vector3f dstLineDir = dstLine.direction().normalized();
        Eigen::Vector3f srcLineDir = initRot * srcLine.direction().normalized();

        Eigen::Vector3f vertLine = srcLineDir.cross(dstLineDir).normalized();
        float distance = (dstLine.middle() - (initRot * srcLine.middle() + initTrans + trans)).dot(vertLine);
        std::cout << i.value() << " --> " << i.key() << ": " << distance << std::endl;
        error += distance;
    }
    error /= pairs.size();

    std::cout << "error: " << error << std::endl;
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

