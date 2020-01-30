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
    pcl::PointCloud<DDBPLineExtractor::MSLPoint>::Ptr firstPointCloud,
    pcl::PointCloud<DDBPLineExtractor::MSL>::Ptr firstLineCloud,
    pcl::PointCloud<DDBPLineExtractor::MSLPoint>::Ptr secondPointCloud,
    pcl::PointCloud<DDBPLineExtractor::MSL>::Ptr secondLineCloud
)
{
    Eigen::Matrix4f finalPose(Eigen::Matrix4f::Identity());
    
    

    return finalPose;
}

Eigen::Quaternionf DDBPLineMatcher::stepRotation(
    float firstDiameter,
    pcl::PointCloud<DDBPLineExtractor::MSLPoint>::Ptr firstPointCloud,
    pcl::PointCloud<DDBPLineExtractor::MSL>::Ptr firstLineCloud,
    float secondDiameter,
    pcl::PointCloud<DDBPLineExtractor::MSLPoint>::Ptr secondPointCloud,
    pcl::PointCloud<DDBPLineExtractor::MSL>::Ptr secondLineCloud,
    pcl::KdTreeFLANN<DDBPLineExtractor::MSLPoint>::Ptr tree,
    float& rotationError,
    float& translationError,
    const Eigen::Quaternionf& initRot,
    QMap<int, int>& pairs)
{
    // 在高帧速下，假设前后帧同一位置的直线位姿变化有限，所以可以通过映射后的直线点云，
    // 直接用kdtree寻找最近的匹配，然后分别计算角度误差和位移误差。先角度后位移。
    float distAvg = 0;
    qDebug() <<"----before----";
    QMap<int, float> errors;
    Eigen::Quaternionf rotAvg(initRot);
    int count = 0;
    pairs.clear();
    for (int i = 0; i < firstPointCloud->size(); i++)
    {
        DDBPLineExtractor::MSLPoint& firstPoint = firstPointCloud->points[i];
        DDBPLineExtractor::MSL& msl1 = firstLineCloud->points[i];

        std::vector<int> indices;
        std::vector<float> distances;
        tree->nearestKSearch(firstPoint, 1, indices, distances);
        Q_ASSERT(indices.size() == 1);

        DDBPLineExtractor::MSLPoint secondPoint = secondPointCloud->points[indices[0]];
        DDBPLineExtractor::MSL msl2 = secondLineCloud->points[indices[0]];

        Eigen::Quaternionf rot = Eigen::Quaternionf::FromTwoVectors(msl1.dir, msl2.dir);

        float angularDistance = rot.angularDistance(Eigen::Quaternionf::Identity());
        float distance = distanceBetweenLines(msl1.dir, msl1.point, msl2.dir, msl2.point);

        if (pairs.contains(indices[0]))
        {
            if (angularDistance > errors[indices[0]])
            {
                continue;
            }
        }

        pairs[indices[0]] = i;
        errors[indices[0]] = angularDistance;

        rotAvg = rotAvg.slerp(1.f / (count + 2), rot);
        distAvg += distance;

        qDebug() << i << "-->" << indices[0] << distances[0] << angularDistance << qRadiansToDegrees(angularDistance) << distance;
        count++;
    }
    distAvg /= firstPointCloud->size();
    qDebug() << qRadiansToDegrees(rotAvg.angularDistance(Eigen::Quaternionf::Identity())) << distAvg;

    for (int i = 0; i < firstPointCloud->size(); i++)
    {
        DDBPLineExtractor::MSLPoint& firstPoint = firstPointCloud->points[i];
        DDBPLineExtractor::MSL& msl1 = firstLineCloud->points[i];

        msl1.dir = rotAvg * msl1.dir;
        msl1.point = rotAvg * msl1.point;
        calculateAlphaBeta(msl1.dir, firstPoint.alpha, firstPoint.beta);
        firstPoint.alpha /= M_PI;
        firstPoint.beta /= M_PI;
    }

    rotationError = rotAvg.angularDistance(Eigen::Quaternionf::Identity());
    translationError = distAvg;

    return rotAvg;
}

Eigen::Vector3f DDBPLineMatcher::stepTranslation(
    float firstDiameter,
    pcl::PointCloud<DDBPLineExtractor::MSLPoint>::Ptr firstPointCloud,
    pcl::PointCloud<DDBPLineExtractor::MSL>::Ptr firstLineCloud,
    float secondDiameter,
    pcl::PointCloud<DDBPLineExtractor::MSLPoint>::Ptr secondPointCloud,
    pcl::PointCloud<DDBPLineExtractor::MSL>::Ptr secondLineCloud,
    pcl::KdTreeFLANN<DDBPLineExtractor::MSLPoint>::Ptr tree,
    float& rotationError,
    float& translationError,
    const Eigen::Quaternionf& initRot,
    const Eigen::Vector3f& initTrans,
        QMap<int, int>& pairs)
{
    float distAvg = 0;
    qDebug() <<"----before----";
    QMap<int, float> errors;
    Eigen::Quaternionf rotAvg(initRot);
    Eigen::Vector3f trans(Eigen::Vector3f::Zero());
    int count = 0;
    pairs.clear();
    for (int i = 0; i < firstPointCloud->size(); i++)
    {
        DDBPLineExtractor::MSLPoint& firstPoint = firstPointCloud->points[i];
        DDBPLineExtractor::MSL& msl1 = firstLineCloud->points[i];

        std::vector<int> indices;
        std::vector<float> distances;
        tree->nearestKSearch(firstPoint, 1, indices, distances);
        Q_ASSERT(indices.size() == 1);

        DDBPLineExtractor::MSLPoint secondPoint = secondPointCloud->points[indices[0]];
        DDBPLineExtractor::MSL msl2 = secondLineCloud->points[indices[0]];

        Eigen::Quaternionf rot = Eigen::Quaternionf::FromTwoVectors(msl1.dir, msl2.dir);

        float angularDistance = rot.angularDistance(Eigen::Quaternionf::Identity());
        float distance = distanceBetweenLines(msl1.dir, msl1.point, msl2.dir, msl2.point);

        if (pairs.contains(indices[0]))
        {
            if (angularDistance > errors[indices[0]])
            {
                continue;
            }
        }

        Eigen::Vector3f cross12 = msl1.dir.cross(msl2.dir);
        Eigen::Vector3f cross21 = msl2.dir.cross(msl1.dir);
        float t1 = (msl2.point - msl1.point).cross(msl2.dir).dot(cross21) / cross21.squaredNorm();
        float t2 = (msl1.point - msl2.point).cross(msl1.dir).dot(cross12) / cross12.squaredNorm();
        Eigen::Vector3f point1 = msl1.point + msl1.dir * t1;
        Eigen::Vector3f point2 = msl2.point + msl2.dir * t2;
        Eigen::Vector3f diff = point2 - point1;

        trans += diff;

        pairs[indices[0]] = i;
        errors[indices[0]] = angularDistance;

        rotAvg = rotAvg.slerp(1.f / (count + 2), rot);
        distAvg += distance;

        qDebug() << i << "-->" << indices[0] << qRadiansToDegrees(angularDistance) << distance << diff.x() << diff.y() << diff.z();
        count++;
    }
    distAvg /= firstPointCloud->size();
    trans /= count;
    rotationError = rotAvg.angularDistance(Eigen::Quaternionf::Identity());
    translationError = distAvg;
    qDebug() << qRadiansToDegrees(rotAvg.angularDistance(Eigen::Quaternionf::Identity())) << distAvg << trans.x() << trans.y() << trans.z();

    for (int i = 0; i < firstPointCloud->size(); i++)
    {
        DDBPLineExtractor::MSLPoint& firstPoint = firstPointCloud->points[i];
        DDBPLineExtractor::MSL& msl1 = firstLineCloud->points[i];

        msl1.point = msl1.point + trans;
        calculateAlphaBeta(msl1.dir, firstPoint.alpha, firstPoint.beta);
        firstPoint.alpha /= M_PI;
        firstPoint.beta /= M_PI;
    }

    return trans;
}

void DDBPLineMatcher::generateDescriptors(pcl::PointCloud<DDBPLineExtractor::MSL>::Ptr& lineCloud, 
    pcl::PointCloud<LineDescriptor>::Ptr& descriptors,
    QList<LineChain>& chains)
{
    chains.clear();
    descriptors.reset(new pcl::PointCloud<LineDescriptor>);
    for (int i1 = 0; i1 < lineCloud->size(); i1++)
    {
        DDBPLineExtractor::MSL msl1 = lineCloud->points[i1];

        int otherIndex = -1;
        float minCos = 1;
        for (int i2 = i1 + 1; i2 < lineCloud->size(); i2++)
        {
            if (i1 == i2)
                continue;

            DDBPLineExtractor::MSL msl2 = lineCloud->points[i2];
            float cos = qAbs(msl1.dir.dot(msl2.dir));
            /*if (cos < minCos)
            {
                minCos = cos;
                otherIndex = i2;
            }*/

            if (cos > 0.9f)
                continue;

            LineChain lc;
            lc.line1 = i1;
            //lc.line2 = otherIndex;
            lc.line2 = i2;
            //DDBPLineExtractor::MSL msl2 = lineCloud->points[otherIndex];

            // 建立局部坐标系
            lc.xLocal = msl1.dir;
            lc.yLocal = msl1.dir.cross(msl2.dir).normalized();
            lc.zLocal = lc.xLocal.cross(lc.yLocal).normalized();

            Eigen::Vector3f cross = msl1.dir.cross(msl2.dir);
            float sqrNorm = cross.squaredNorm();
            float t1 = (msl2.point - msl1.point).cross(msl2.dir).dot(cross) / sqrNorm;
            float t2 = (msl1.point - msl2.point).cross(msl1.dir).dot(cross) / sqrNorm;
            lc.point1 = msl1.point + msl1.dir * t1;
            lc.point2 = msl2.point + msl2.dir * t2;

            //LineDescriptor descriptor;
            //for (int i3 = 0; i3 < lineCloud->size(); i3++)
            //{
            //    DDBPLineExtractor::MSL msl = lineCloud->points[i3];

            //    float cosY = msl.dir.dot(lc.yLocal);
            //    float radiansY = qAcos(cosY);
            //    //float radiusX = qAcos((msl.dir - lc.yLocal * cosY).normalized().dot(lc.xLocal));
            //    float radiansX = qAcos(msl.dir.cross(lc.xLocal).dot(lc.xLocal));
            //    int x = static_cast<int>(radiansX / (M_PI / LINE_MATCHER_DIVISION));
            //    int y = static_cast<int>(radiansY / (M_PI / LINE_MATCHER_DIVISION));
            //    int dim = y * LINE_MATCHER_DIVISION + x;
            //    descriptor.elems[dim] = descriptor.elems[dim] + 1;
            //}
            //QString line;
            //for (int i = 0; i < LineDescriptor::elemsSize(); i++)
            //{
            //    line.append(QString::number(descriptor.elems[i]));
            //    line.append(" ");
            //}
            //qDebug() << i1 << otherIndex << minCos << chains.size() << "  " << line;
            //descriptors->points.push_back(descriptor);
            chains.append(lc);
        }
    }

    for (int i = 0; i < chains.size(); i++)
    {
        LineChain& lc1 = chains[i];
        Eigen::Vector3f origin = (lc1.point1 + lc1.point2) / 2;
        DDBPLineExtractor::MSL msl1 = lineCloud->points[lc1.line1];

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

        float distTick = maxDistance / (LINE_MATCHER_DIVISION - 1);
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
            int dim2 = LINE_MATCHER_DIVISION * LINE_MATCHER_DIVISION + static_cast<int>(dir.norm() / distTick);
            descriptor.elems[dim] = descriptor.elems[dim] + 1;
            descriptor.elems[dim2] = descriptor.elems[dim2] + 1;
        }
        QString line;
        for (int i = 0; i < LineDescriptor::elemsSize(); i++)
        {
            line.append(QString::number(descriptor.elems[i]));
            line.append(" ");
        }
        qDebug() << i << lc1.name() << ":" << line << maxDistance;
        descriptors->points.push_back(descriptor);
    }
    qDebug() << "end";
}

