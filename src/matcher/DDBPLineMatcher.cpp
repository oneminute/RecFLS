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

Eigen::Matrix4f DDBPLineMatcher::stepRotation(
    float firstDiameter,
    pcl::PointCloud<DDBPLineExtractor::MSLPoint>::Ptr firstPointCloud, 
    pcl::PointCloud<DDBPLineExtractor::MSL>::Ptr firstLineCloud, 
    float secondDiameter,
    pcl::PointCloud<DDBPLineExtractor::MSLPoint>::Ptr secondPointCloud, 
    pcl::PointCloud<DDBPLineExtractor::MSL>::Ptr secondLineCloud, 
    const Eigen::Matrix4f& initPose)
{
    //// 建立Line-Chain
    //generateDescriptors(firstLineCloud, m_descriptors1, m_chains1);
    //generateDescriptors(secondLineCloud, m_descriptors2, m_chains2);

    //m_descMat1.resize(m_descriptors1->size(), LINE_MATCHER_ELEMDIMS);
    //for (int i = 0; i < m_descriptors1->size(); i++)
    //{
    //    for (int j = 0; j < LineDescriptor::elemsSize(); j++)
    //    {
    //        m_descMat1.row(i)[j] = m_descriptors1->points[i].elems[j];
    //        m_descMat1.row(i).normalize();
    //    }
    //}

    //m_descMat2.resize(m_descriptors2->size(), LINE_MATCHER_ELEMDIMS);
    //for (int i = 0; i < m_descriptors2->size(); i++)
    //{
    //    for (int j = 0; j < LineDescriptor::elemsSize(); j++)
    //    {
    //        m_descMat2.row(i)[j] = m_descriptors2->points[i].elems[j];
    //        m_descMat2.row(i).normalize();
    //    }
    //}
    ////m_descMat1.normalize();
    ////m_descMat2.normalize();

    //qDebug() << "chains1 size:" << m_chains1.size() << ", chains2 size:" << m_chains2.size();

    //Eigen::MatrixXf result = m_descMat1 * m_descMat2.transpose();
    //for (int i = 0; i < result.rows(); i++)
    //{
    //    int otherIndex;
    //    float maxValue = result.row(i).maxCoeff(&otherIndex);

    //    qDebug().nospace().noquote() << i << "(" << m_chains1[i].line1 << "," << m_chains1[i].line2 << ") --> " << otherIndex << "(" << m_chains2[otherIndex].line1 << "," << m_chains2[otherIndex].line2 << ") " << maxValue;
    //}

    //qDebug() << "------------------";

    //for (int i = 0; i < m_descMat1.rows(); i++)
    //{
    //    float minDistance = std::numeric_limits<float>::max();
    //    int otherIndex = 0;
    //    for (int j = 0; j < m_descMat2.rows(); j++)
    //    {
    //        float dist = (m_descMat1.row(i) - m_descMat2.row(j)).norm();
    //        if (dist < minDistance)
    //        {
    //            minDistance = dist;
    //            otherIndex = j;
    //        }
    //    }
    //    qDebug().nospace().noquote() << i << "(" << m_chains1[i].line1 << "," << m_chains1[i].line2 << ") --> " << otherIndex << "(" << m_chains2[otherIndex].line1 << "," << m_chains2[otherIndex].line2 << ") " << minDistance;
    //}

    //qDebug() << "------------------";

    //pcl::KdTreeFLANN<LineDescriptor> descTree;
    //descTree.setInputCloud(m_descriptors2);

    //for (int i = 0; i < m_descriptors1->size(); i++)
    //{
    //    std::vector<int> indices;
    //    std::vector<float> distances;
    //    indices.resize(2);
    //    distances.resize(2);
    //    descTree.nearestKSearch(m_descriptors1->points[i], 2, indices, distances);
    //    qDebug() << "chain" << i << "-->" << indices[0] << distances[0];
    //}

    pcl::KdTreeFLANN<DDBPLineExtractor::MSLPoint> tree;
    tree.setInputCloud(secondPointCloud);

    for (int it = 0; it < 10; it++)
    {
        // 在高帧速下，假设前后帧同一位置的直线位姿变化有限，所以可以通过映射后的直线点云，
        // 直接用kdtree寻找最近的匹配，然后分别计算角度误差和位移误差。先角度后位移。
        float distAvg = 0;
        qDebug() << it << "----before----";
        QMap<int, int> pairs;
        QMap<int, float> errors;
        Eigen::Quaternionf rotAvg(Eigen::Quaternionf::Identity());
        Eigen::Vector3f trans(Eigen::Vector3f::Zero());
        for (int i = 0; i < firstPointCloud->size(); i++)
        {
            DDBPLineExtractor::MSLPoint firstPoint = firstPointCloud->points[i];
            std::vector<int> indices;
            std::vector<float> distances;
            tree.nearestKSearch(firstPoint, 1, indices, distances);
            Q_ASSERT(indices.size() == 1);

            DDBPLineExtractor::MSLPoint secondPoint = secondPointCloud->points[indices[0]];

            DDBPLineExtractor::MSL msl1 = firstLineCloud->points[i];
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

            rotAvg = rotAvg.slerp(1.f / (i + 2), rot);
            distAvg += distance;

            //float radians = qAcos(msl1.dir.dot(msl2.dir));

            qDebug() << i << "-->" << indices[0] << distances[0] << angularDistance << qRadiansToDegrees(angularDistance) << distance;
        }
        distAvg /= firstPointCloud->size();
        qDebug() << qRadiansToDegrees(rotAvg.angularDistance(Eigen::Quaternionf::Identity())) << distAvg;
        qDebug() << it << "----after----";

        for (int i = 0; i < firstPointCloud->size(); i++)
        {
            DDBPLineExtractor::MSLPoint& firstPoint = firstPointCloud->points[i];
            DDBPLineExtractor::MSL& msl1 = firstLineCloud->points[i];
            Eigen::Vector3f dir1 = rotAvg * msl1.dir;
            //Eigen::Vector3f point1 = rotAvg * msl1.point;
            msl1.dir = dir1;
            //msl1.point = point1;
            qDebug() << "+" << firstPoint.alpha << firstPoint.beta;
            calculateAlphaBeta(dir1, firstPoint.alpha, firstPoint.beta);
            firstPoint.alpha /= M_PI;
            firstPoint.beta /= M_PI;
            qDebug() << "-" << firstPoint.alpha << firstPoint.beta;
            //firstPoint.x = point1.x() / firstDiameter;
            //firstPoint.y = point1.y() / firstDiameter;
            //firstPoint.z = point1.z() / firstDiameter;
        }

        Eigen::Quaternionf rotAvg2(Eigen::Quaternionf::Identity());
        distAvg = 0;
        int count = 0;
        for (QMap<int, int>::iterator i = pairs.begin(); i != pairs.end(); i++, count++)
        {
            DDBPLineExtractor::MSLPoint firstPoint = firstPointCloud->points[i.value()];
            DDBPLineExtractor::MSLPoint secondPoint = firstPointCloud->points[i.key()];

            DDBPLineExtractor::MSL msl1 = firstLineCloud->points[i.value()];
            DDBPLineExtractor::MSL msl2 = secondLineCloud->points[i.key()];

            Eigen::Quaternionf rot = Eigen::Quaternionf::FromTwoVectors(msl1.dir, msl2.dir);

            float angularDistance = rot.angularDistance(Eigen::Quaternionf::Identity());
            float distance = distanceBetweenLines(msl1.dir, msl1.point, msl2.dir, msl2.point);

            rotAvg2 = rotAvg2.slerp(1.f / (count + 2), rot);
            distAvg += distance;

            qDebug() << i.value() << "-->" << i.key() << angularDistance << qRadiansToDegrees(angularDistance) << distance;
        }
        distAvg /= count;
        qDebug() << qRadiansToDegrees(rotAvg2.angularDistance(Eigen::Quaternionf::Identity())) << distAvg;
    }
    return Eigen::Matrix4f();
}

Eigen::Matrix4f DDBPLineMatcher::stepTranslate(pcl::PointCloud<DDBPLineExtractor::MSLPoint>::Ptr firstPointCloud, pcl::PointCloud<DDBPLineExtractor::MSL>::Ptr firstLineCloud, pcl::PointCloud<DDBPLineExtractor::MSLPoint>::Ptr secondPointCloud, pcl::PointCloud<DDBPLineExtractor::MSL>::Ptr secondLineCloud, const Eigen::Matrix4f& initPose)
{
    return Eigen::Matrix4f();
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

