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
    QList<LineChain>& chains1,
    pcl::PointCloud<MSL>::Ptr& lines1,
    pcl::PointCloud<LineDescriptor2>::Ptr& desc1,
    QList<LineChain>& chains2,
    pcl::PointCloud<MSL>::Ptr& lines2,
    pcl::PointCloud<LineDescriptor2>::Ptr& desc2
)
{
    Eigen::Matrix4f finalPose(Eigen::Matrix4f::Identity());

    qDebug() << "------------------"; 
    qDebug() << "desc1 size:" << desc1->size() << "desc2 size:" << desc2->size();

    pcl::KdTreeFLANN<LineDescriptor2> descTree; 
    descTree.setInputCloud(desc2); 
    Eigen::Matrix3f rOut(Eigen::Matrix3f::Identity());
    Eigen::Vector3f tOut(Eigen::Vector3f::Zero());
    float errorOutT = 0;
    float errorOutR = 0;
    float errorOut = 0;

    QMap<int, float> coefs;
    m_pairs.clear();
    for (int i = 0; i < desc1->size(); i++)
    {
        std::vector<int> indices;
        std::vector<float> distances;
        descTree.nearestKSearch(desc1->points[i], 2, indices, distances);

        LineChain lc1 = chains1[i];
        LineChain& lc2 = chains2[indices[0]];
        qDebug() << i << lc1.name() << "-->" << indices[0] << lc2.name() << ", distance:" << distances[0];

        float angleDiff11 = qAbs(qAcos(lc1.line1.dir.dot(lc2.line1.dir)));
        float angleDiff12 = qAbs(qAcos(lc1.line2.dir.dot(lc2.line2.dir)));

        qDebug() << i << "-->" << indices[0] << qRadiansToDegrees(qAbs(angleDiff11 - angleDiff12)) << qRadiansToDegrees(angleDiff11) << qRadiansToDegrees(angleDiff12);

        if (qAbs(angleDiff11 - angleDiff12) >= (M_PI / 4))
            continue;

        if (angleDiff11 > (M_PI / 20))
            continue;
        if (angleDiff12 > (M_PI / 20))
            continue;

        //float angleDiff21 = qAbs(qAcos(msl11.dir.dot(msl22.dir)));
        //float angleDiff22 = qAbs(qAcos(msl12.dir.dot(msl21.dir)));

        float dist = (lc1.point - lc2.point).norm();
        if (dist >= 0.1f)
            continue;

        qDebug() << i << "-->" << indices[0] << ": distance:" << distances[0];

        if (m_pairs.contains(indices[0]))
        {
            if (distances[0] >= coefs[indices[0]])
                continue;
        }

        m_pairs[indices[0]] = i;
        coefs[indices[0]] = distances[0];
    }

    m_pairIndices = m_pairs.keys();
    qSort(m_pairIndices.begin(), m_pairIndices.end(), [=](int a, int b) -> bool
        {
            return coefs[a] < coefs[b];
        }
    );

    int totalIterations = 1;
    for (int iteration = 0; iteration < totalIterations; iteration++)
    {
        // 使用SVD求解，H是前后帧两个直线方向向量样本集合的协方差矩阵
        Eigen::Matrix3f H(Eigen::Matrix3f::Zero());
        Eigen::Vector3f dirAvg1(Eigen::Vector3f::Zero());
        Eigen::Vector3f dirAvg2(Eigen::Vector3f::Zero());
        Eigen::VectorXf weights;
        weights.resize(m_pairIndices.size() * 2);
        for (int i = 0; i < m_pairIndices.size(); i++)
        {
            int index2 = m_pairIndices[i];
            int index1 = m_pairs[index2];

            LineChain lc1 = chains1[index1];
            LineChain lc2 = chains2[index2];

            dirAvg1 += lc1.line1.dir;
            dirAvg1 += lc1.line2.dir;
            dirAvg2 += lc2.line1.dir;
            dirAvg2 += lc2.line2.dir;

            //W.col(i * 2)[i * 2] = W.col(i * 2 + 1)[i * 2 + 1] = 1.f / m_pairIndices.size();
            //weights[i * 2] = weights[i * 2 + 1] = 1 / coefs[index2];
            weights[i * 2] = weights[i * 2 + 1] = 1 - coefs[index2];
            //weights[i * 2] = weights[i * 2 + 1] = -qLn(coefs[index2]);

            qDebug() << "inlines:" << index1 << "-->" << index2 << coefs[index2];
        }
        dirAvg1 /= (m_pairIndices.size() * 2);
        dirAvg2 /= (m_pairIndices.size() * 2);
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
        X.resize(3, m_pairIndices.size() * 2);
        Y.resize(3, m_pairIndices.size() * 2);
        for (int i = 0; i < m_pairIndices.size(); i++)
        {
            int index2 = m_pairIndices[i];
            int index1 = m_pairs[index2];

            LineChain lc1 = chains1[index1];
            LineChain lc2 = chains2[index2];

            Eigen::Vector3f dir11 = lc1.line1.dir - dirAvg1;
            Eigen::Vector3f dir12 = lc1.line2.dir - dirAvg1;
            Eigen::Vector3f dir21 = lc2.line1.dir - dirAvg2;
            Eigen::Vector3f dir22 = lc2.line2.dir - dirAvg2;

            X.col(i * 2) = dir11;
            X.col(i * 2 + 1) = dir12;
            Y.col(i * 2) = dir21;
            Y.col(i * 2 + 1) = dir22;

            p1 += lc1.point * weights[i * 2] * 2;
            p2 += lc2.point * weights[i * 2 + 1] * 2;
            //p1 += lc1.point;
            //p2 += lc2.point;
        }
        std::cout << "X:" << std::endl << X << std::endl;
        std::cout << "Y:" << std::endl << Y << std::endl;
        H = X * W * Y.transpose();
        std::cout << "H:" << std::endl << H << std::endl;
        //p1 = p1 / m_pairIndices.size();
        //p2 = p2 / m_pairIndices.size();

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

        rOut = R * rOut ;
        tOut += t;

        // 更新数据
        for (int i = 0; i < lines1->size(); i++)
        {
            lines1->points[i].dir = R * lines1->points[i].dir;
            lines1->points[i].point = R * lines1->points[i].point + t;
        }
        for (int i = 0; i < m_pairIndices.size(); i++)
        {
            int index2 = m_pairIndices[i];
            int index1 = m_pairs[index2];

            LineChain& lc1 = chains1[index1];
            LineChain& lc2 = chains2[index2];

            lc1.line1.dir = R * lc1.line1.dir;
            lc1.line1.point = R * lc1.line1.point + t;
            lc1.line2.dir = R * lc1.line2.dir;
            lc1.line2.point = R * lc1.line2.point + t;
            lc1.point = R * lc1.point + t;
            lc1.point1 = R * lc1.point1 + t;
            lc1.point2 = R * lc1.point2 + t;
            lc1.xLocal = R * lc1.xLocal;
            lc1.yLocal = R * lc1.yLocal;
            lc1.zLocal = R * lc1.zLocal;

            float errorT = 0;
            float errorR = 0;

            errorT = (lc2.point - lc1.point).norm();
            //errorR += (lc1.line1.dir - lc2.line1.dir).norm();
            //errorR += (lc1.line2.dir - lc2.line2.dir).norm();
            errorR += qAbs(qAcos(lc1.line1.dir.dot(lc2.line1.dir)));
            errorR += qAbs(qAcos(lc1.line2.dir.dot(lc2.line2.dir)));

            errorOutT += errorT;
            errorOutR += errorR;

            qDebug() << index1 << "-->" << index2 << ": errorT:" << errorT << ", errorR:" << errorR;
        }
    }

    errorOutT /= (totalIterations * m_pairIndices.size());
    errorOutR /= (totalIterations * m_pairIndices.size() * 2);
    errorOut = errorOutT + errorOutR;

    finalPose.topLeftCorner(3, 3) = rOut;
    finalPose.topRightCorner(3, 1) = tOut;
    std::cout << "finalPose:" << std::endl << finalPose << std::endl;
    qDebug() << "error:" << errorOut;
    return finalPose;
}

Eigen::Quaternionf LineMatcher::stepRotation(
    float firstDiameter,
    pcl::PointCloud<MSLPoint>::Ptr firstPointCloud,
    pcl::PointCloud<MSL>::Ptr firstLineCloud,
    float secondDiameter,
    pcl::PointCloud<MSLPoint>::Ptr secondPointCloud,
    pcl::PointCloud<MSL>::Ptr secondLineCloud,
    pcl::KdTreeFLANN<MSLPoint>::Ptr tree,
    float& rotationError,
    float& translationError,
    QMap<int, int>& pairs)
{
    // 在高帧速下，假设前后帧同一位置的直线位姿变化有限，所以可以通过映射后的直线点云，
    // 直接用kdtree寻找最近的匹配，然后分别计算角度误差和位移误差。先角度后位移。
    float distAvg = 0;
    qDebug() <<"- - - - rotation - - - -";
    //qDebug() << initRot.x() << initRot.y() << initRot.z() << initRot.w();
    QMap<int, float> errors;
    QMap<int, Eigen::Quaternionf> rots;
    QMap<int, float> dists;
    Eigen::Quaternionf rotAvg(Eigen::Quaternionf::Identity());
    int count = 0;
    pairs.clear();
    for (int i = 0; i < firstPointCloud->size(); i++)
    {
        MSLPoint firstPoint = firstPointCloud->points[i];
        MSL msl1 = firstLineCloud->points[i];

        std::vector<int> indices;
        std::vector<float> distances;
        tree->nearestKSearch(firstPoint, 1, indices, distances);
        Q_ASSERT(indices.size() == 1);

        MSLPoint secondPoint = secondPointCloud->points[indices[0]];
        MSL msl2 = secondLineCloud->points[indices[0]];

        Eigen::Quaternionf rot = Eigen::Quaternionf::FromTwoVectors(msl1.dir, msl2.dir);

        float angularDistance = rot.angularDistance(Eigen::Quaternionf::Identity());
        float distance = distanceBetweenLines(msl1.dir, msl1.point, msl2.dir, msl2.point);

        if (pairs.contains(indices[0]))
        {
            if (distances[0] > errors[indices[0]] || distance >= 0.2f)
            {
                continue;
            }
        }

        rots.insert(indices[0], rot);
        pairs[indices[0]] = i;
        errors[indices[0]] = distances[0];
        dists[indices[0]] = distance;

        rotAvg = rotAvg.slerp(1.f / (count + 1), rot);

        qDebug() << i << "-->" << indices[0] << distances[0] << angularDistance << qRadiansToDegrees(angularDistance) << distance;
        count++;
    }
    float errorAvg = rotAvg.angularDistance(Eigen::Quaternionf::Identity());

    // 方差过滤
    Eigen::Quaternionf rotOut(Eigen::Quaternionf::Identity());
    count = 0;
    QList<int> removal;
    Eigen::MatrixXf a(pairs.size(), 3);
    Eigen::MatrixXf b(pairs.size(), 3);
    for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end();)
    {
        float error = errors[i.key()];
        float variance = 1 / qPow(error - errorAvg, 2);
        qDebug().noquote() << i.value() << "-->" << i.key() << variance;
        MSL msl1 = firstLineCloud->points[i.value()];
        MSL msl2 = secondLineCloud->points[i.key()];
        a.row(count) = msl1.dir;
        b.row(count) = msl2.dir;
        if (variance < 500)
        {
            removal.append(i.key());
            //i = pairs.erase(i);
            //continue;
        }

        Eigen::Quaternionf rot = rots[i.key()];
        rotOut = rotOut.slerp(1.f / (count + 1), rot);
        distAvg += dists[i.key()];
        count++;
        i++;
    }
    distAvg /= count;
    //Eigen::Vector3f sol = a.colPivHouseholderQr().solve(b);
    Eigen::Matrix3f sol = (a.transpose() * a).inverse() * a.transpose() * b;
    Eigen::Quaternionf q(sol);
    qDebug() << "sol = " << q.x() << q.y() << q.z() << qRadiansToDegrees(q.angularDistance(Eigen::Quaternionf::Identity()));

    qDebug() << "removal pairs:" << removal;
    rotationError = rotOut.angularDistance(Eigen::Quaternionf::Identity());
    translationError = distAvg;
    qDebug() << rotationError << qRadiansToDegrees(rotationError) << translationError;
    qDebug() << rotOut.x() << rotOut.y() << rotOut.z() << rotOut.w() << "|" << rotAvg.x() << rotAvg.y() << rotAvg.z() << rotAvg.w();

    if (pairs.isEmpty())
    {
        rotOut = Eigen::Quaternionf::Identity();
        return rotOut;
    }

    for (int i = 0; i < firstPointCloud->size(); i++)
    {
        MSLPoint& firstPoint = firstPointCloud->points[i];
        MSL& msl1 = firstLineCloud->points[i];

        msl1.dir = rotOut * msl1.dir;
        msl1.point = rotOut * msl1.point;
        calculateAlphaBeta(msl1.dir, firstPoint.alpha, firstPoint.beta);
        firstPoint.alpha /= M_PI;
        firstPoint.beta /= M_PI;
    }

    return rotOut;
}

Eigen::Vector3f LineMatcher::stepTranslation(
    pcl::PointCloud<MSL>::Ptr firstLineCloud,
    pcl::PointCloud<MSL>::Ptr secondLineCloud,
    pcl::KdTreeFLANN<MSLPoint>::Ptr tree,
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
        MSL msl1 = firstLineCloud->points[i.value()];
        MSL msl2 = secondLineCloud->points[i.key()];

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

                MSL& msl = firstLineCloud->points[pairs[keys[j]]];
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
        MSL& msl1 = firstLineCloud->points[index1];
        MSL& msl2 = secondLineCloud->points[index2];

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
            MSL& msl = firstLineCloud->points[pairs[keys[j]]];
            msl.point += diff;
        }
    }

    for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end(); i++)
    {
        int index2 = i.key();
        int index1 = i.value();
        MSL& msl1 = firstLineCloud->points[index1];
        MSL& msl2 = secondLineCloud->points[index2];

        float distance = 0;
        Eigen::Vector3f diff = transBetweenLines(msl1.dir, msl1.point, msl2.dir, msl2.point, distance);

        distAvg += distance;
    }

    distAvg /= pairs.size();
    translationError = distAvg;
    qDebug().noquote() << distAvg << "[" << trans.x() << trans.y() << trans.z() << "]";

    return trans;
}

