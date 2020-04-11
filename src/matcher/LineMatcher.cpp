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

Eigen::Matrix3f LineMatcher::stepRotation(
    pcl::PointCloud<Line>::Ptr lines1,
    pcl::PointCloud<Line>::Ptr lines2,
    pcl::KdTreeFLANN<Line>::Ptr tree,
    float& rotationError,
    float& translationError,
    QMap<int, int>& pairs)
{
    // 在高帧速下，假设前后帧同一位置的直线位姿变化有限，所以可以通过映射后的直线点云，
    // 直接用kdtree寻找最近的匹配，然后分别计算角度误差和位移误差。先角度后位移。
    float distAvg = 0;
    Eigen::Matrix3f rotOut(Eigen::Matrix3f::Identity());
    qDebug() <<"- - - - rotation - - - -";
    //qDebug() << initRot.x() << initRot.y() << initRot.z() << initRot.w();
    QMap<int, float> errors;
    QMap<int, Eigen::Quaternionf> rots;
    QMap<int, float> dists;
    Eigen::Quaternionf rotAvg(Eigen::Quaternionf::Identity());
    int count = 0;
    pairs.clear();
    for (int i = 0; i < lines1->size(); i++)
    {
        Line line1 = lines1->points[i];

        std::vector<int> indices;
        std::vector<float> distances;
        tree->nearestKSearch(line1, 1, indices, distances);
        Q_ASSERT(indices.size() == 1);

        Line line2 = lines2->points[indices[0]];

        Eigen::Quaternionf rot = Eigen::Quaternionf::FromTwoVectors(line1.dir, line2.dir);

        if (pairs.contains(indices[0]))
        {
            if (distances[0] > errors[indices[0]])
            {
                continue;
            }
        }

        pairs[indices[0]] = i;
        errors[indices[0]] = distances[0];

        //qDebug() << i << "-->" << indices[0] << distances[0];
        count++;
    }

    count = 0;
    /*for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end(); i++)
    {
        int key = i.key();
        int value = i.value();
        qDebug().noquote() << key << "-->" << value;
    }*/

    // 使用SVD求解，H是前后帧两个直线方向向量样本集合的协方差矩阵
    Eigen::Matrix3f H(Eigen::Matrix3f::Zero());
    Eigen::Vector3f dirAvg1(Eigen::Vector3f::Zero());
    Eigen::Vector3f dirAvg2(Eigen::Vector3f::Zero());
    Eigen::VectorXf weights;
    weights.resize(pairs.size());
    count = 0;
    for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end(); i++, count++)
    {
        int index2 = i.key();
        int index1 = i.value();
        qDebug().noquote() << index1 << "-->" << index2;

        dirAvg1 += lines1->points[index1].dir;
        dirAvg2 += lines2->points[index2].dir;

        weights[count]  = 1.f / pairs.size();
    }
    dirAvg1 /= (pairs.size());
    dirAvg2 /= (pairs.size());
    std::cout << "dirAvg1:" << dirAvg1.transpose() << std::endl;
    std::cout << "dirAvg2:" << dirAvg2.transpose() << std::endl;

    weights /= weights.sum();
    // W是各样本的权值主对角矩阵，它的迹应为1
    Eigen::MatrixXf W(weights.asDiagonal());
    std::cout << "W:" << std::endl << W << std::endl;

    Eigen::Vector3f p1(Eigen::Vector3f::Zero());
    Eigen::Vector3f p2(Eigen::Vector3f::Zero());
    Eigen::MatrixXf X;
    Eigen::MatrixXf Y;
    X.resize(3, pairs.size());
    Y.resize(3, pairs.size());
    count = 0;
    for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end(); i++, count++)
    {
        int index2 = i.key();
        int index1 = i.value();

        Eigen::Vector3f dir1 = lines1->points[index1].dir - dirAvg1;
        Eigen::Vector3f dir2 = lines2->points[index2].dir - dirAvg1;

        X.col(count) = dir1;
        Y.col(count) = dir2;

        //p1 += lc1.point * weights[i * 2] * 2;
        //p2 += lc2.point * weights[i * 2 + 1] * 2;
        //p1 += lc1.point;
        //p2 += lc2.point;
    }
    std::cout << "X:" << std::endl << X << std::endl;
    std::cout << "Y:" << std::endl << Y << std::endl;
    H = X * W * Y.transpose();
    std::cout << "H:" << std::endl << H << std::endl;
    //p1 = p1 / pairs.size();
    //p2 = p2 / pairs.size();

    Eigen::JacobiSVD<Eigen::MatrixXf> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
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
    qDebug() << "det =" << det;
    std::cout << "V:" << std::endl << V << std::endl;
    std::cout << "S:" << std::endl << S << std::endl;
    std::cout << "U:" << std::endl << U << std::endl;
    Eigen::Matrix3f R = V * S * U.transpose();

    //Eigen::Vector3f t = posAvg1 - R * posAvg2;
    Eigen::Vector3f t = p2 - R * p1;

    std::cout << "         p1: " << p1.transpose() << std::endl;
    std::cout << "         p2: " << p2.transpose() << std::endl;
    std::cout << "R * posAvg1: " << (R * p1).transpose() << std::endl;

    std::cout << "R:" << std::endl;
    std::cout << R << std::endl;
    std::cout << "t: " << t.transpose() << std::endl;

    rotOut = R * rotOut ;

    // 更新数据
    
    return rotOut;
}

Eigen::Vector3f LineMatcher::stepTranslation(
    pcl::PointCloud<Line>::Ptr firstLineCloud,
    pcl::PointCloud<Line>::Ptr secondLineCloud,
    pcl::KdTreeFLANN<Line>::Ptr tree,
    float& translationError,
    QMap<int, int>& pairs)
{
    qDebug() <<"- - - - translation - - - -";
    QMap<int, float> errors;
    float distAvg = 0;
    Eigen::Vector3f trans(Eigen::Vector3f::Zero());
    QList<int> keys;
    for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end(); i++)
    {
        Line msl1 = firstLineCloud->points[i.value()];
        Line msl2 = secondLineCloud->points[i.key()];

        float cos = qAcos(msl1.dir.dot(msl2.dir));

        errors[i.key()] = cos;
        keys.append(i.key());
    }

    qSort(keys.begin(), keys.end(), [=](int v1, int v2) -> bool 
        {
            return errors[v1] < errors[v2];
        }
    );

    QSet<int> processed;
    int lastIndex = -1;
    for (int i = 0; i < keys.size(); i++)
    {
        Eigen::Vector3f diff(Eigen::Vector3f::Zero());
        Eigen::Vector3f lastDir(Eigen::Vector3f::Zero());
        int index = -1;
        if (i == 0)
        {
            float dist;
            lastDir = transBetweenLines(firstLineCloud->points[pairs[keys[0]]].dir, firstLineCloud->points[pairs[keys[0]]].point, 
                secondLineCloud->points[keys[0]].dir, secondLineCloud->points[keys[0]].point, dist);
            index = 0;
        }
        else
        {
            lastDir = (firstLineCloud->points[pairs[keys[lastIndex]]].dir + secondLineCloud->points[keys[lastIndex]].dir) / 2;
            if (lastDir.dot(secondLineCloud->points[keys[lastIndex]].point - firstLineCloud->points[pairs[keys[lastIndex]]].point) < 0)
            {
                lastDir = -lastDir;
            }

            float minCos = 1;
            for (int j = 0; j < keys.size(); j++)
            {
                if (processed.contains(keys[j]))
                    continue;

                Line& msl = firstLineCloud->points[pairs[keys[j]]];
                float cos = qAbs(msl.dir.dot(lastDir));
                if (cos >= 0.95f)
                {
                    continue;
                }

                if (cos < minCos)
                {
                    index = j;
                    minCos = cos;
                }
            }

            if (index < 0)
            {
                break;
            }
        }
        lastDir.normalize();
        processed.insert(keys[index]);

        lastIndex = index;
        int index2 = keys[index];
        int index1 = pairs[index2];
        qDebug().nospace().noquote() << " ++++ index" << index << ". " << index1 << " --> " << index2 << ", error = " << errors[index2];
        Line& msl1 = firstLineCloud->points[index1];
        Line& msl2 = secondLineCloud->points[index2];

        float distance = 0;
        Eigen::Vector3f lineDiff = transBetweenLines(msl1.dir, msl1.point, msl2.dir, msl2.point, distance);
        diff = lastDir * (distance / lineDiff.normalized().dot(lastDir));
        if (diff.dot(msl2.point - msl1.point) < 0)
        {
            diff = -diff;
        }
        qDebug() << "  diff: [" << diff.x() << diff.y() << diff.z() << "]";
        qDebug() << "  lineDiff: [" << lineDiff.x() << lineDiff.y() << lineDiff.z() << "]";
        qDebug() << "  distance:" << distance << ", projected distance:" << (distance / lineDiff.normalized().dot(lastDir));
        if (qIsNaN(diff.x()) || qIsNaN(diff.y()) || qIsNaN(diff.z()) || qIsInf(diff.x()) || qIsInf(diff.y()) || qIsInf(diff.z()))
        {
            continue;
        }

        trans += diff;

        for (int j = 0; j < keys.size(); j++)
        {
            Line& msl = firstLineCloud->points[pairs[keys[j]]];
            msl.point += diff;
        }
    }

    for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end(); i++)
    {
        int index2 = i.key();
        int index1 = i.value();
        Line& msl1 = firstLineCloud->points[index1];
        Line& msl2 = secondLineCloud->points[index2];

        float distance = 0;
        Eigen::Vector3f diff = transBetweenLines(msl1.dir, msl1.point, msl2.dir, msl2.point, distance);

        distAvg += distance;
    }

    distAvg /= pairs.size();
    translationError = distAvg;
    qDebug().noquote() << distAvg << "[" << trans.x() << trans.y() << trans.z() << "]";

    return trans;
}

