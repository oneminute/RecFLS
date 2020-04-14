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
    , PROPERTY_INIT(MaxIterations, 30)
{

}

Eigen::Matrix4f LineMatcher::compute(pcl::PointCloud<Line>::Ptr lines1, pcl::PointCloud<Line>::Ptr lines2, float& rotationError, float& translationError)
{
    Eigen::Matrix4f out = Eigen::Matrix4f::Identity();
    pcl::KdTreeFLANN<Line>::Ptr tree(new pcl::KdTreeFLANN<Line>());
    tree->setInputCloud(lines2);
    QMap<int, int> pairs;
    for (int i = 0; i < MaxIterations(); i++)
    {
        Eigen::Matrix4f stepM = step(lines1, lines2, tree, rotationError, translationError, pairs);
        out = stepM * out;
    }
    return out;
}

Eigen::Matrix4f LineMatcher::step(pcl::PointCloud<Line>::Ptr lines1, pcl::PointCloud<Line>::Ptr lines2, pcl::KdTreeFLANN<Line>::Ptr tree, float& rotationError, float& translationError, QMap<int, int>& pairs)
{
    Eigen::Matrix4f M = Eigen::Matrix4f::Identity();
    M.topLeftCorner(3, 3) = stepRotation(lines1, lines2, tree, pairs);
    M.topRightCorner(3, 1) = stepTranslation(lines1, lines2, tree, pairs);

    rotationError = 0;
    translationError = 0;

    for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end(); i++)
    {
        Line line1 = lines1->points[i.value()];
        Line line2 = lines2->points[i.key()];

        float transError = qAbs(distanceBetweenLines(line1.dir, line1.point, line2.dir, line2.point));
        float rotError = qAbs(qAcos(line1.dir.dot(line2.dir)));

        translationError += transError;
        rotationError += rotError;
    }
    translationError /= pairs.size();
    rotationError /= pairs.size();

    return M;
}

Eigen::Matrix3f LineMatcher::stepRotation(
    pcl::PointCloud<Line>::Ptr lines1,
    pcl::PointCloud<Line>::Ptr lines2,
    pcl::KdTreeFLANN<Line>::Ptr tree,
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
        line1.debugPrint();

        std::vector<int> indices;
        std::vector<float> distances;
        tree->nearestKSearch(line1, 1, indices, distances);
        Q_ASSERT(indices.size() == 1);

        Line line2 = lines2->points[indices[0]];

        Eigen::Quaternionf rot = Eigen::Quaternionf::FromTwoVectors(line1.dir, line2.dir);

        if (distances[0] > 0.1f)
            continue;

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
        qDebug().noquote() << index1 << "-->" << index2 << errors[index1];

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
    }
    std::cout << "X:" << std::endl << X << std::endl;
    std::cout << "Y:" << std::endl << Y << std::endl;
    H = X * W * Y.transpose();
    std::cout << "H:" << std::endl << H << std::endl;

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

    Eigen::Vector3f t = p2 - R * p1;

    std::cout << "         p1: " << p1.transpose() << std::endl;
    std::cout << "         p2: " << p2.transpose() << std::endl;
    std::cout << "R * posAvg1: " << (R * p1).transpose() << std::endl;

    std::cout << "R:" << std::endl;
    std::cout << R << std::endl;
    std::cout << "t: " << t.transpose() << std::endl;

    rotOut = R * rotOut ;

    // 更新数据
    for (int i = 0; i < lines1->size(); i++)
    {
        Line& line1 = lines1->points[i];

        line1.dir = R * line1.dir;
        Eigen::Vector3f point = R * line1.point;
        line1.point = closedPointOnLine(point, line1.dir, line1.point);

        line1.generateDescriptor();
    }
    
    return rotOut;
}

Eigen::Vector3f LineMatcher::stepTranslation(
    pcl::PointCloud<Line>::Ptr lines1,
    pcl::PointCloud<Line>::Ptr lines2,
    pcl::KdTreeFLANN<Line>::Ptr tree,
    QMap<int, int>& pairs)
{
    qDebug() <<"- - - - translation - - - -";
    QMap<int, float> errors;
    float distAvg = 0;
    Eigen::Vector3f trans(Eigen::Vector3f::Zero());
    QList<int> keys;
    for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end(); i++)
    {
        Line line1 = lines1->points[i.value()];
        Line line2 = lines2->points[i.key()];

        float dist = 0;
        Eigen::Vector3f t = transBetweenLines(line1.dir, line1.point, line2.dir, line2.point, dist);

        std::cout << i.key() << " --> " << i.value() << t.transpose() << std::endl;

        trans += t;
    }

    trans /= pairs.size();

    // 更新数据
    for (int i = 0; i < lines1->size(); i++)
    {
        Line& line1 = lines1->points[i];

        line1.point = line1.point + trans;
        line1.generateDescriptor();
    }

    std::cout << trans.transpose() << std::endl;

    return trans;
}

