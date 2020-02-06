#include "DDBPLineMatcher.h"

#include <QDebug>
#include <QtMath>

#include <pcl/common/pca.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>

#include "util/Utils.h"

DDBPLineMatcher::DDBPLineMatcher(QObject* parent)
    : QObject(parent)
{

}

Eigen::Matrix4f DDBPLineMatcher::compute(
    pcl::PointCloud<MSLPoint>::Ptr firstPointCloud,
    pcl::PointCloud<MSL>::Ptr firstLineCloud,
    pcl::PointCloud<MSLPoint>::Ptr secondPointCloud,
    pcl::PointCloud<MSL>::Ptr secondLineCloud
)
{
    Eigen::Matrix4f finalPose(Eigen::Matrix4f::Identity());

    // 生成Line-Chain 
    generateDescriptors(firstLineCloud, m_descriptors1, m_chains1); 
    generateDescriptors(secondLineCloud, m_descriptors2, m_chains2); 

    m_descMat1.resize(m_descriptors1->size(), LINE_MATCHER_ELEMDIMS); 
    for (int i = 0; i < m_descriptors1->size(); i++) 
    { 
        for (int j = 0; j < LineDescriptor::elemsSize(); j++) 
        { 
            m_descMat1.row(i)[j] = m_descriptors1->points[i].elems[j]; 
            m_descMat1.row(i).normalize(); 
        } 
    } 

    m_descMat2.resize(m_descriptors2->size(), LINE_MATCHER_ELEMDIMS); 
    for (int i = 0; i < m_descriptors2->size(); i++) 
    { 
        for (int j = 0; j < LineDescriptor::elemsSize(); j++) 
        { 
            m_descMat2.row(i)[j] = m_descriptors2->points[i].elems[j]; 
            m_descMat2.row(i).normalize(); 
        } 
    } 
    //m_descMat1.normalize(); 
    //m_descMat2.normalize(); 

    qDebug() << "chains1 size:" << m_chains1.size() << ", chains2 size:" << m_chains2.size(); 

    Eigen::MatrixXf result = m_descMat1 * m_descMat2.transpose(); 
    for (int i = 0; i < result.rows(); i++) 
    { 
        int otherIndex; 
        float maxValue = result.row(i).maxCoeff(&otherIndex); 

        qDebug().nospace().noquote() << i << "(" << m_chains1[i].line1 << "," << m_chains1[i].line2 << ") --> " << otherIndex << "(" << m_chains2[otherIndex].line1 << "," << m_chains2[otherIndex].line2 << ") " << maxValue; 
    } 

    qDebug() << "------------------"; 

    for (int i = 0; i < m_descMat1.rows(); i++) 
    { 
        float minDistance = std::numeric_limits<float>::max(); 
        int otherIndex = 0; 
        for (int j = 0; j < m_descMat2.rows(); j++) 
        { 
            float dist = (m_descMat1.row(i) - m_descMat2.row(j)).norm(); 
            if (dist < minDistance) 
            { 
                minDistance = dist; 
                otherIndex = j; 
            } 
        } 
        qDebug().nospace().noquote() << i << "(" << m_chains1[i].line1 << "," << m_chains1[i].line2 << ") --> " << otherIndex << "(" << m_chains2[otherIndex].line1 << "," << m_chains2[otherIndex].line2 << ") " << minDistance; 
    } 

    qDebug() << "------------------"; 

    pcl::KdTreeFLANN<LineDescriptor> descTree; 
    descTree.setInputCloud(m_descriptors2); 

    for (int i = 0; i < m_descriptors1->size(); i++) 
    { 
        std::vector<int> indices; 
        std::vector<float> distances; 
        indices.resize(2); 
        distances.resize(2); 
        descTree.nearestKSearch(m_descriptors1->points[i], 2, indices, distances); 
        qDebug() << "chain" << i << "-->" << indices[0] << distances[0]; 
    } 

    return finalPose;
}

Eigen::Quaternionf DDBPLineMatcher::stepRotation(
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

        if (pairs.contains(indices[0]) || distance >= 0.2f)
        {
            if (distances[0] > errors[indices[0]])
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

Eigen::Vector3f DDBPLineMatcher::stepTranslation(
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
            lastDir = transBetweenLines(firstLineCloud->points[pairs[keys[0]]].dir, firstLineCloud->points[pairs[keys[0]]].point, secondLineCloud->points[keys[0]].dir, secondLineCloud->points[keys[0]].point, dist);
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

void DDBPLineMatcher::generateDescriptors(pcl::PointCloud<MSL>::Ptr& lineCloud, 
    pcl::PointCloud<LineDescriptor>::Ptr& descriptors,
    QList<LineChain>& chains)
{
    chains.clear();
    descriptors.reset(new pcl::PointCloud<LineDescriptor>);
    for (int i1 = 0; i1 < lineCloud->size(); i1++)
    {
        MSL msl1 = lineCloud->points[i1];

        int otherIndex = -1;
        float minCos = 1;
        for (int i2 = 0; i2 < lineCloud->size(); i2++)
        {
            if (i1 == i2)
                continue;

            MSL msl2 = lineCloud->points[i2];
            float cos = qAbs(msl1.dir.dot(msl2.dir));
            if (cos > 0.9f)
                continue;

            LineChain lc;
            lc.line1 = i1;
            lc.line2 = i2;

            // 建立局部坐标系
            lc.xLocal = msl1.dir;
            lc.yLocal = msl1.dir.cross(msl2.dir).normalized();
            lc.zLocal = lc.xLocal.cross(lc.yLocal).normalized();

            Eigen::Vector3f cross12 = msl1.dir.cross(msl2.dir);
            Eigen::Vector3f cross21 = msl2.dir.cross(msl1.dir);
            float t1 = (msl2.point - msl1.point).cross(msl2.dir).dot(cross21) / cross21.squaredNorm();
            float t2 = (msl1.point - msl2.point).cross(msl1.dir).dot(cross12) / cross12.squaredNorm();
            lc.point1 = msl1.point + msl1.dir * t1;
            lc.point2 = msl2.point + msl2.dir * t2;

            chains.append(lc);
        }
    }

    qDebug() << "chains size:" << chains.size();

    for (int i = 0; i < chains.size(); i++)
    {
        LineChain& lc1 = chains[i];
        Eigen::Vector3f origin = (lc1.point1 + lc1.point2) / 2;
        MSL msl1 = lineCloud->points[lc1.line1];

        float maxDistance = 0;
        for (int j = 0; j < chains.size(); j++)
        {
            if (i == j)
                continue;

            LineChain& lc2 = chains[j];
            Eigen::Vector3f middle = (lc2.point1 + lc2.point2) / 2;
            float distance = (middle - origin).norm();

            if (distance > maxDistance)
            {
                maxDistance = distance;
            }
        }

        float distTick = maxDistance / (LINE_MATCHER_DIST_DIVISION - 1);
        LineDescriptor descriptor;
        for (int j = 0; j < chains.size(); j++)
        {
            if (i == j)
                continue;

            LineChain& lc2 = chains[j];
            Eigen::Vector3f middle = (lc2.point1 + lc2.point2) / 2;

            Eigen::Vector3f dir = middle - origin;
            float cosY = lc2.yLocal.dot(lc1.yLocal);
            float radiansY = qAcos(cosY);
            float radiansX = qAcos(lc2.yLocal.cross(lc1.yLocal).dot(lc1.zLocal));
            int x = static_cast<int>(radiansX / (M_PI / (LINE_MATCHER_DIVISION - 1)));
            int y = static_cast<int>(radiansY / (M_PI / (LINE_MATCHER_DIVISION - 1)));
            int dim = y * LINE_MATCHER_DIVISION + x;
            int dim2 = LINE_MATCHER_ANGLE_ELEMDIMS + static_cast<int>(dir.norm() / distTick);
            descriptor.elems[dim] = descriptor.elems[dim] + 1;
            descriptor.elems[dim2] = descriptor.elems[dim2] + 2;
        }
        QString line;
        for (int i = 0; i < LineDescriptor::elemsSize(); i++)
        {
            line.append(QString::number(descriptor.elems[i]));
            line.append(" ");
        }
        qDebug().noquote() << i << lc1.name() << ":" << line << maxDistance;
        descriptors->points.push_back(descriptor);
    }
    qDebug() << "end";
}

