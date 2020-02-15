#include "LineExtractor.h"

#include <QtMath>
#include <QDebug>
#include <QVector>

#include <pcl/common/pca.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/kdtree_flann.hpp>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>

#include "util/Utils.h"

LineExtractor::LineExtractor(QObject* parent)
    : QObject(parent)
    , m_angleMappingMethod(TWO_DIMS)
    , m_searchRadius(0.05f)
    , m_minNeighbours(3)
    , m_searchErrorThreshold(0.025f)
    , m_angleSearchRadius(qDegreesToRadians(20.0) * M_1_PI)
    , m_angleMinNeighbours(10)
    , m_mappingTolerance(0.01f)
    , m_regionGrowingZDistanceThreshold(0.005f)
    , m_minLineLength(0.01f)
    , m_mslRadiusSearch(0.01f)
    , m_boundBoxDiameter(2)
{

}

QList<LineSegment> LineExtractor::compute(const pcl::PointCloud<pcl::PointXYZI>::Ptr& boundaryCloud)
{
    qDebug() << "angleMappingMethod" << m_angleMappingMethod;
    qDebug() << "searchRadius:" << m_searchRadius;
    qDebug() << "minNeighbours:" << m_minNeighbours;
    qDebug() << "searchErrorThreshold:" << m_searchErrorThreshold;
    qDebug() << "angleSearchRadius:" << m_angleSearchRadius;
    qDebug() << "angleMinNeighbours:" << m_angleMinNeighbours;
    qDebug() << "mappingTolerance:" << m_mappingTolerance;
    qDebug() << "z distance threshold:" << m_regionGrowingZDistanceThreshold;
    qDebug() << "minLineLength:" << m_minLineLength;
    qDebug() << "msl radius search:" << m_mslRadiusSearch;

    m_boundaryCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    m_angleCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    m_dirCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    m_mappingCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    m_centerCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    m_linedCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    m_boundaryIndices.reset(new std::vector<int>);
    m_angleCloudIndices.clear();
    m_density.clear();
    m_offsetRate.clear();
    m_errors.clear();
    m_linePointsCount.clear();

    QList<LineSegment> lines;

    // 计算当前点云外包盒的最大直径，该值将用于对直线长度进行归一化的分母。
    Eigen::Vector4f minPoint, maxPoint;
    pcl::getMinMax3D<pcl::PointXYZI>(*boundaryCloud, minPoint, maxPoint);
    m_minPoint = minPoint.head(3);
    m_maxPoint = maxPoint.head(3);
    m_boundBoxDiameter = (m_maxPoint - m_minPoint).norm();
    qDebug() << "bound box diameter:" << m_boundBoxDiameter << minPoint.x() << minPoint.y() << minPoint.z() << maxPoint.x() << maxPoint.y() << maxPoint.z() << (m_maxPoint - m_minPoint).norm();

    pcl::search::KdTree<pcl::PointXYZI> tree;
    tree.setInputCloud(boundaryCloud);

    // globalDir用于统一所有的直线方向
    Eigen::Vector3f globalDir(1, 1, 1);
    globalDir.normalize();
    Eigen::Vector3f xAxis(1, 0, 0);
    Eigen::Vector3f yAxis(0, 1, 0);
    Eigen::Vector3f zAxis(0, 0, 1);

    // 计算每一个边界点的PCA主方向，
    int index = 0;
    for (int i = 0; i < boundaryCloud->size(); i++)
    {
        pcl::PointXYZI ptIn = boundaryCloud->points[i];
        Eigen::Vector3f point = ptIn.getVector3fMap();

        // 查找近邻
        std::vector<int> neighbourIndices;
        std::vector<float> neighbourDistances;
        tree.radiusSearch(ptIn, m_searchRadius, neighbourIndices, neighbourDistances);

        // 近邻点太少表示这是一个离群点，且小于3也无法进行PCA计算
        if (neighbourIndices.size() < m_minNeighbours)
        {
            continue;
        }

        // PCA计算当前点的主方向
        pcl::PCA<pcl::PointXYZI> pca;
        pca.setInputCloud(boundaryCloud);
        pca.setIndices(pcl::IndicesPtr(new std::vector<int>(neighbourIndices)));
        Eigen::Vector3f eigenValues = pca.getEigenValues();
        Eigen::Vector3f::Index maxIndex;
        eigenValues.maxCoeff(&maxIndex);

        // 主方向
        Eigen::Vector3f primeDir = pca.getEigenVectors().col(maxIndex).normalized();
        // 中点
        Eigen::Vector3f meanPoint = pca.getMean().head(3);

        // 计算误差
        float error = 0;
        for (int j = 0; j < neighbourIndices.size(); j++)
        {
            int neighbourIndex = neighbourIndices[j];
            pcl::PointXYZI ptNeighbour = boundaryCloud->points[neighbourIndex];
            // 求垂直距离
            float vertDist = primeDir.cross(ptNeighbour.getVector3fMap() - meanPoint).norm();
            error += vertDist;
        }
        error /= neighbourIndices.size();

        //qDebug() << i << "error =" << error << (error < m_searchErrorThreshold);
        // 如果误差值太大，则跳过
        if (error >= m_searchErrorThreshold)
        {
            continue;
        }

        // 保证与全局方向同向
        if (primeDir.dot(globalDir) < 0)
        {
            primeDir = -primeDir;
        }

        Eigen::Vector3f eulerAngles(Eigen::Vector3f::Zero());
        // 使用二维或三维的映射角。二维使用计算出的俯仰角和航向角
        if (m_angleMappingMethod == TWO_DIMS)
        {
            float alpha, beta;
            // 计算俯仰角和水平角
            calculateAlphaBeta(primeDir, alpha, beta);

            eulerAngles.x() = alpha * M_1_PI;
            eulerAngles.y() = beta * M_1_PI;
        }
        else if (m_angleMappingMethod = THREE_DIMS)
        {
            // 计算欧拉角
            Eigen::Quaternionf rotQuat = Eigen::Quaternionf();
            rotQuat.setFromTwoVectors(globalDir, primeDir);
            Eigen::Matrix3f rotMatrix = rotQuat.toRotationMatrix();
            eulerAngles = rotMatrix.eulerAngles(0, 1, 2) * M_1_PI;
        }

        // 计算当前点到穿过原点的主方向所在直线的垂直距离
        float distance = point.cross(primeDir).norm();

        // 计算当前点投影到穿过原点的主方向所在直线的投影点的有向长度
        Eigen::Vector3f pointProjToPrimeDir = primeDir * point.dot(primeDir);
        float xCoord = pointProjToPrimeDir.norm();
        if (pointProjToPrimeDir.dot(primeDir) < 0)
        {
            xCoord = -xCoord;
        }

        // 当前点与投影到穿过原点的主方向直线的投影点所形成的直线，计算该直线与深度方向的角度
        Eigen::Vector3f lineVertProj = (point - pointProjToPrimeDir).normalized();
        float cosZ = lineVertProj.dot(zAxis);
        float radiansZ = qAcos(cosZ);
        if (lineVertProj.cross(zAxis).dot(primeDir) < 0)
        {
            radiansZ = (M_PI * 2) - radiansZ;
        }

        // 建立点主方向角度值点云
        pcl::PointXYZI anglePt;
        anglePt.x = eulerAngles.x();
        anglePt.y = eulerAngles.y();
        anglePt.z = eulerAngles.z();
        anglePt.intensity = index;
        m_angleCloud->push_back(anglePt);
        
        // 建立主方向点云，主要是为了保留主方向，供后续计算使用。
        pcl::PointXYZI dirPt;
        dirPt.getVector3fMap() = primeDir;
        dirPt.intensity = index;
        m_dirCloud->push_back(dirPt);

        // 用主方向原点垂距、主方向垂面夹角和主方向单维距离这三个数据形成一个新的映射点云，用于使用聚集抽取连续的直线。
        pcl::PointXYZI mappingPt;
        mappingPt.x = distance / m_boundBoxDiameter * 2;
        mappingPt.y = radiansZ / (M_PI * 2);
        mappingPt.z = xCoord / m_boundBoxDiameter * 2;
        mappingPt.intensity = index;
        m_mappingCloud->push_back(mappingPt);

        // 将计算出的质心单独保存成一个点云，方便后续计算。
        pcl::PointXYZI centerPt;
        centerPt.getVector3fMap() = meanPoint;
        centerPt.intensity = index;
        m_centerCloud->push_back(centerPt);

        // 将计算出的点云密度单独保存成一个点云，方便后续计算。
        pcl::PointXYZI pt = ptIn;
        pt.intensity = 0;
        m_boundaryIndices->push_back(index);
        m_boundaryCloud->push_back(ptIn);

        /*if (index % 100 == 0)
        {
            qDebug() << eulerAngles.x() << eulerAngles.y() << eulerAngles.z();
        }*/
        m_angleCloudIndices.append(index);
        index++;
    }
    m_boundaryCloud->width = m_boundaryCloud->points.size();
    m_boundaryCloud->height = 1;
    m_boundaryCloud->is_dense = true;
    m_angleCloud->width = m_angleCloud->points.size();
    m_angleCloud->height = 1;
    m_angleCloud->is_dense = true;
    m_dirCloud->width = m_dirCloud->points.size();
    m_dirCloud->height = 1;
    m_dirCloud->is_dense = true;
    m_mappingCloud->width = m_mappingCloud->points.size();
    m_mappingCloud->height = 1;
    m_mappingCloud->is_dense = true;
    m_centerCloud->width = m_centerCloud->points.size();
    m_centerCloud->height = 1;
    m_centerCloud->is_dense = true;

    qDebug() << "Input cloud size:" << boundaryCloud->size() << ", filtered cloud size:" << m_boundaryCloud->size();

    // 创建参数化点云的查找树，该树用于通过角度值点云来查找主方向相同的聚集
    pcl::search::KdTree<pcl::PointXYZI> angleCloudTree;
    angleCloudTree.setInputCloud(m_angleCloud);

    // 下面这个循环用于查找主方向相同的点，但并不使用区域增长，否则会产生较大的偏离
    //float maxDensity = 0;
    //float minDensity = m_angleCloud->size();
    QMap<int, std::vector<int>> neighbours;
    for (int i = 0; i < m_angleCloud->size(); i++)
    {
        pcl::PointXYZI ptAngle = m_angleCloud->points[i];

        // 查找近邻
        std::vector<int> neighbourIndices;
        std::vector<float> neighbourDistances;
        angleCloudTree.radiusSearch(ptAngle, m_angleSearchRadius, neighbourIndices, neighbourDistances);
        neighbours.insert(i, neighbourIndices);
        
        // 计算密度值
        float density = neighbourIndices.size();

        /*if (density > maxDensity)
        {
            maxDensity = density;
        }
        if (density < minDensity)
        {
            minDensity = density;
        }*/

        m_density.append(density);

        // 计算质心值
        Eigen::Vector3f center(Eigen::Vector3f::Zero());
        for (int n = 0; n < neighbourIndices.size(); n++)
        {
            int neighbourIndex = neighbourIndices[n];
            pcl::PointXYZI ptNeighbour = m_angleCloud->points[neighbourIndex];
            center += ptNeighbour.getVector3fMap();
        }
        center /= density;

        // 计算离心率
        float offsetRate = (ptAngle.getVector3fMap() - center).norm();
        m_offsetRate.append(offsetRate);
    }

    // 根据密度值和离心率值对所有的映射点进行排序。
    qSort(m_angleCloudIndices.begin(), m_angleCloudIndices.end(), [=](int v1, int v2) -> bool
        {
            if (v1 >= m_angleCloudIndices.size() || v2 >= m_angleCloudIndices.size())
                return false;

            if (m_density[v1] == m_density[v2])
            {
                return m_offsetRate[v1] < m_offsetRate[v2];
            }
            else
            {
                return m_density[v1] > m_density[v2];
            }
        }
    );

    // 从密度最大的值开始挑出每一个映射点，每一个映射点及其近邻点即为一组同方向的点，但这些点的方向
    // 可能仅仅是平行而相距很远。
    QVector<bool> processed(m_angleCloud->size(), false);
    for (int i = 0; i < m_angleCloudIndices.size(); i++)
    {
        int index = m_angleCloudIndices[i];
        if (processed[index])
        {
            continue;
        }
        //qDebug() << indexList[i] << m_density[indexList[i]];

        pcl::PointXYZI ptAngle = m_angleCloud->points[index];
        std::vector<int> neighbourIndices = neighbours[index];
        if (neighbourIndices.size() >= m_angleMinNeighbours)
        {
            std::vector<int> subCloudIndices;
            for (int n = 0; n < neighbourIndices.size(); n++)
            {
                int neighbourIndex = neighbourIndices[n];
                if (processed[neighbourIndex])
                {
                    continue;
                }
                subCloudIndices.push_back(static_cast<int>(m_angleCloud->points[neighbourIndex].intensity));
                processed[neighbourIndex] = true;
            }

            // 如果当前映射点的近邻数量足够，则再次使用条件化欧式聚集来对这组点再次聚集，以找出共线且连续的线段。
            if (subCloudIndices.size() >= m_angleMinNeighbours)
            {
                m_subCloudIndices.insert(m_angleCloud->points[index].intensity, subCloudIndices);

                qDebug() << "index:" << index << ", sub cloud size:" << subCloudIndices.size();

                pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
                tree->setInputCloud(m_mappingCloud);

                std::vector<pcl::PointIndices> clusterIndices;

                // 使用条件化欧式聚集来进行区域增长，重点在Z方向的增长，而限制在XY方向的增长。
                pcl::ConditionalEuclideanClustering<pcl::PointXYZI> cec(true);
                cec.setConditionFunction([=](const pcl::PointXYZI& ptSeed, const pcl::PointXYZI& ptCandidate, float sqrDistance) -> bool
                    {
                        if (qSqrt(sqrDistance) < m_mappingTolerance)
                        {
                            Eigen::Vector3f seedPoint = ptSeed.getVector3fMap();
                            Eigen::Vector3f candidatePoint = ptCandidate.getVector3fMap();

                            float distToZ = (candidatePoint - seedPoint).cross(zAxis).norm();
                            if (distToZ < m_regionGrowingZDistanceThreshold)
                            {
                                return true;
                            }
                        }
                        return false;
                    }
                );
                cec.setClusterTolerance(m_mappingTolerance);
                cec.setMinClusterSize(1);
                cec.setMaxClusterSize(subCloudIndices.size());
                cec.setSearchMethod(tree);
                cec.setInputCloud(m_mappingCloud);
                cec.setIndices(pcl::IndicesPtr(new std::vector<int>(subCloudIndices)));
                cec.segment(clusterIndices);

                qDebug() << "  cluster size:" << clusterIndices.size();
                std::vector<pcl::PointIndices>::iterator itClusters = clusterIndices.begin();
                while (itClusters != clusterIndices.end())
                {
                    pcl::PointIndices indices = *itClusters;
                    Eigen::Vector3f dir(0, 0, 0);
                    Eigen::Vector3f start(0, 0, 0);
                    Eigen::Vector3f end(0, 0, 0);
                    Eigen::Vector3f center(0, 0, 0);

                    if (indices.indices.size() >= m_angleMinNeighbours)
                    {
                        // 计算全体近邻点方向的平均的方向作为主方向，同时计算质点。
                        for (std::vector<int>::const_iterator itIndices = indices.indices.begin(); itIndices != indices.indices.end(); ++itIndices)
                        {
                            int localIndex = m_mappingCloud->points[*itIndices].intensity;
                            pcl::PointXYZI ptBoundary = m_boundaryCloud->points[localIndex];
                            pcl::PointXYZI ptDir = m_dirCloud->points[localIndex];
                            dir += ptDir.getVector3fMap();
                            center += ptBoundary.getVector3fMap();

                            ptBoundary.intensity = index;
                            m_linedCloud->push_back(ptBoundary);
                        }
                        dir /= indices.indices.size();
                        center /= indices.indices.size();

                        // 验证直线，计算每个点到目标直线的距离期望，使用该值作为误差以验证是否为有效的直线。
                        // 同时计算这一组点形成的起点与终点。
                        float error = 0;
                       
                        for (std::vector<int>::const_iterator itIndices = indices.indices.begin(); itIndices != indices.indices.end(); ++itIndices)
                        {
                            int index = m_mappingCloud->points[*itIndices].intensity;
                            pcl::PointXYZI ptBoundary = m_boundaryCloud->points[index];
                            Eigen::Vector3f boundaryPoint = ptBoundary.getVector3fMap();
                            error = (boundaryPoint - center).cross(dir).norm();

                            if (start.isZero())
                            {
                                // 如果第一次循环，让当前点作为起点
                                start = closedPointOnLine(boundaryPoint, dir, center);
                            }
                            else
                            {
                                // 将当前点与当前计算出的临时起点连在一起，查看其与当前聚集主方向的一致性，若一致则当前点为新的起点。
                                if ((start - boundaryPoint).dot(dir) > 0)
                                {
                                    start = closedPointOnLine(boundaryPoint, dir, center);
                                }
                            }

                            if (end.isZero())
                            {
                                // 如果第一次循环，让当前点作为终点
                                end = closedPointOnLine(boundaryPoint, dir, center);
                            }
                            else
                            {
                                // 将当前点与当前计算出的临时终点连在一起，查看其与当前聚集主方向的一致性，若一致则当前点为新的终点。
                                if ((boundaryPoint - end).dot(dir) > 0)
                                {
                                    end = closedPointOnLine(boundaryPoint, dir, center);
                                }
                            }
                        }
                        error /= indices.indices.size();
                        //qDebug() << "    error:" << error;

                        LineSegment ls(start, end);
                        if (ls.length() > m_minLineLength)
                        {
                            lines.append(ls);
                            m_errors.append(error);
                            m_linePointsCount.append(indices.indices.size());
                        }
                    }
                    itClusters++;
                }
            }
        }
    }
    m_linedCloud->width = m_linedCloud->points.size();
    m_linedCloud->height = 1;
    m_linedCloud->is_dense = true;
    qDebug() << "Extract" << m_subCloudIndices.size() << "groups.";

    // 再次对线段进行映射，每一个线段映射为一个5维向量。
    // 其中前两个维度为俯仰角和航向角，后三个维度是原点到直线垂线交点的位置。
    m_mslPointCloud.reset(new pcl::PointCloud<MSLPoint>);
    m_mslCloud.reset(new pcl::PointCloud<MSL>);
    QList<bool> mslProcessed;
    QList<int> mslIndices;
    for (int i = 0; i < lines.size(); i++)
    {
        MSLPoint mslPoint;
        LineSegment line = lines[i];
        float alpha, beta;
        Eigen::Vector3f dir = line.direction().normalized();
        calculateAlphaBeta(dir, alpha, beta);
        mslPoint.alpha = alpha / M_PI;
        mslPoint.beta = beta / M_PI;
        Eigen::Vector3f start = line.start();
        Eigen::Vector3f closedPoint = closedPointOnLine(Eigen::Vector3f::Zero(), dir, start);
        mslPoint.x = closedPoint.x() / m_boundBoxDiameter;
        mslPoint.y = closedPoint.y() / m_boundBoxDiameter;
        mslPoint.z = closedPoint.z() / m_boundBoxDiameter;
        m_mslPointCloud->push_back(mslPoint);
        mslProcessed.append(false);
        mslIndices.append(i);

        MSL msl;
        msl.dir = dir;
        msl.point = closedPoint;
        msl.weight = 1;
        m_mslCloud->push_back(msl);
    }

    /*pcl::KdTreeFLANN<MSLPoint> mslTree;
    mslTree.setInputCloud(m_mslPointCloud);

    for (int i = 0; i < m_mslPointCloud->size(); i++)
    {
        MSLPoint mslPoint = m_mslPointCloud->points[mslIndices[i]];
        if (mslProcessed[i])
            continue;

        std::vector<int> indices;
        std::vector<float> distances;
        mslTree.radiusSearch(mslPoint, m_mslRadiusSearch, indices, distances);

        float maxLineLength = 0;
        Eigen::Vector3f dir;
        Eigen::Vector3f point;
        for (int j = 0; j < indices.size(); j++)
        {
            int neighbourIndex = indices[j];
            MSLPoint neighbour = m_mslPointCloud->points[neighbourIndex];
            mslProcessed[neighbourIndex] = true;
            LineSegment line = lines[neighbourIndex];
            if (line.length() > maxLineLength)
            {
                maxLineLength = line.length();
                dir = line.direction().normalized();
                point.x() = neighbour.x * m_boundBoxDiameter;
                point.y() = neighbour.y * m_boundBoxDiameter;
                point.z() = neighbour.z * m_boundBoxDiameter;
            }
        }

        MSL msl;
        msl.dir = dir;
        msl.point = point;
        msl.weight = maxLineLength / m_boundBoxDiameter;
        m_mslCloud->push_back(msl);
        qDebug() << "msl" << i << ":" << indices.size() << msl.weight;
    }*/

    return lines;
}

void LineExtractor::extractLinesFromPlanes(const QList<Plane>& planes)
{
    //m_mslPointCloud.reset(new pcl::PointCloud<MSLPoint>);
    //m_mslCloud.reset(new pcl::PointCloud<MSL>);
    Eigen::Vector3f globalDir(1, 1, 1);

    for (int i = 0; i < planes.size(); i++)
    {
        Plane plane1 = planes[i];
        for (int j = i + 1; j < planes.size(); j++)
        {
            //if (i == j)
                //continue;

            Plane plane2 = planes[j];

            float cos = plane1.dir.dot(plane2.dir);
            if (qAbs(qRadiansToDegrees(qAcos(cos))) < 30.f)
                continue;

            Eigen::Vector3f crossLine = plane1.dir.cross(plane2.dir).normalized();
            if (crossLine.dot(globalDir) < 0)
                crossLine = -crossLine;
            if (crossLine.isZero())
                continue;

            // 求得平面1上的点到交线的垂线方向
            Eigen::Vector3f lineOnPlane1 = plane1.dir.cross(crossLine).normalized();
            lineOnPlane1.normalize();
            float dist1 = (plane2.point - plane1.point).dot(plane2.dir) / (lineOnPlane1.dot(plane2.dir));
            // 求得平面1上的点在交线上投影的交点
            Eigen::Vector3f crossPoint1 = plane1.point + lineOnPlane1 * dist1;

            //Eigen::Vector3f lineOnPlane2 = plane2.dir.cross(crossLine).normalized();
            //lineOnPlane2.normalize();
            //float dist2 = (plane1.point - plane2.point).dot(plane1.dir) / (lineOnPlane2.dot(plane1.dir));
            //Eigen::Vector3f crossPoint2 = plane2.point + lineOnPlane2 * dist2;

            //Eigen::Vector3f crossPoint = (crossPoint1 + crossPoint2) / 2;
            //Eigen::Vector3f crossPoint = crossPoint1;
            Eigen::Vector3f closedPoint = closedPointOnLine(Eigen::Vector3f::Zero(), crossLine, crossPoint1);
            
            //Eigen::Vector3f point = closedPointOnLine(plane1.point, line);
            MSLPoint mslPoint;
            float alpha, beta;
            calculateAlphaBeta(crossLine, alpha, beta);
            mslPoint.alpha = alpha / M_PI;
            mslPoint.beta = beta / M_PI;
            mslPoint.x = closedPoint.x() / m_boundBoxDiameter;
            mslPoint.y = closedPoint.y() / m_boundBoxDiameter;
            mslPoint.z = closedPoint.z() / m_boundBoxDiameter;
            m_mslPointCloud->push_back(mslPoint);

            MSL msl;
            msl.dir = crossLine;
            msl.point = closedPoint;
            msl.weight = 1;
            m_mslCloud->push_back(msl);
        }
    }

    m_mslPointCloud->width = m_mslPointCloud->points.size();
    m_mslPointCloud->height = 1;
    m_mslPointCloud->is_dense = true;
    m_mslCloud->width = m_mslCloud->points.size();
    m_mslCloud->height = 1;
    m_mslCloud->is_dense = true;
}

void LineExtractor::generateLineChains()
{
    m_chains.clear();

    // 计算所有直线方向的协方差矩阵
    Eigen::VectorXf meanDir;
    Eigen::MatrixXf mat;
    mat.resize(3, m_mslCloud->size());
    for (int i = 0; i < m_mslCloud->size(); i++)
    {
        mat.col(i) = m_mslCloud->points[i].dir;
        //avgDir += m_mslCloud->points[i].dir;
    }
    meanDir = mat.rowwise().mean();
    //Eigen::RowVectorXf meanVecRow(Eigen::RowVectorXf::Map(meanDir.data(), mat.rows()));
    //Eigen::MatrixXf zeroMeanMat = mat;
    //zeroMeanMat.colwise() -= meanDir;
    //Eigen::MatrixXf covMat = (zeroMeanMat * zeroMeanMat.adjoint()) / double(mat.cols() - 1);
    //Eigen::MatrixXf covMatInv = covMat.inverse();
    //qDebug() << "meanVecRow size:" << meanVecRow.cols() << meanVecRow.rows();
    //std::cout << "mat\n" << mat << std::endl;
    //std::cout << "meanDir\n" << meanDir << std::endl;
    //std::cout << "meanVecRow\n" << meanVecRow << std::endl;
    //std::cout << "zeroMeanMat\n" << zeroMeanMat << std::endl;
    //std::cout << "zeroMeanMat.adjoint()\n" << zeroMeanMat.adjoint() << std::endl;
    //std::cout << "covMat\n" << covMat << std::endl;
    //std::cout << "covMatInv\n" << covMatInv << std::endl;
    //qDebug() << covMatInv.col(0).norm() << covMatInv.col(1).norm() << covMatInv.col(2).norm();
 
    m_maxLineChainLength = 0;
    pcl::PointCloud<pcl::PointXYZ>::Ptr middlePointCloud(new pcl::PointCloud<pcl::PointXYZ>);
    for (int i = 0; i < m_mslCloud->size(); i++)
    {
        for (int j = i + 1; j < m_mslCloud->size(); j++)
        {
            MSL msl1 = m_mslCloud->points[i];
            MSL msl2 = m_mslCloud->points[j];

            int index1 = i;
            int index2 = j;

            if (msl2.dir.dot(Eigen::Vector3f::UnitX()) > msl1.dir.dot(Eigen::Vector3f::UnitX()))
            {
                qDebug() << "revert";

                index1 = j;
                index2 = i;
                msl1 = m_mslCloud->points[index1];
                msl2 = m_mslCloud->points[index2];
            }

            float radians = qAbs(qAcos(msl1.dir.dot(msl2.dir)));
            //bool valid = true;
            if (radians < M_PI / 4)
                continue;

            //if (valid)
            //{
            LineChain lc;
            lc.line1 = index1;
            lc.line2 = index2;
            lc.radians = radians;

            // 建立局部坐标系
            lc.xLocal = msl1.dir;
            lc.yLocal = msl1.dir.cross(msl2.dir).normalized();
            lc.zLocal = lc.xLocal.cross(lc.yLocal).normalized();

            // 计算最近点
            Eigen::Vector3f p1;     // 表示直线1上的一个点
            Eigen::Vector3f p2;     // 表示直线2上的一个点
            Eigen::Vector3f c1;     // 表示两条直线的叉乘直线方向，P1-P2
            Eigen::Vector3f c2;     // 表示两条直线的叉乘直线方向，P2-P1
            Eigen::Vector3f d1;     // 表示线1的方向
            Eigen::Vector3f d2;     // 表示线2的方向
            float t1;               // P1到cross1的距离
            float t2;               // P2到cross2的距离
            Eigen::Vector3f cross1; // 线1上的最近点
            Eigen::Vector3f cross2; // 线2上的最近点
            float l;                // 两条直线间的垂线距离

            p1 = msl1.point;
            p2 = msl2.point;
            d1 = msl1.dir.normalized();
            d2 = msl2.dir.normalized();
            c1 = d1.cross(d2).normalized();
            if (c1.dot(p1 - p2) < 0)
                c1 = -c1;
            c2 = -c1;
            l = qAbs((p1 - p2).dot(c1));

            t1 = ((p2 - p1).cross(d2) + c1.cross(d2) * l).dot(d1.cross(d2)) / d1.cross(d2).squaredNorm();
            t2 = ((p1 - p2).cross(d1) + c2.cross(d1) * l).dot(d2.cross(d1)) / d2.cross(d1).squaredNorm();

            cross1 = p1 + d1 * t1;
            cross2 = p2 + d2 * t2;

            lc.point1 = cross1;
            lc.point2 = cross2;
            lc.point = (cross1 + cross2) / 2;
            lc.length = (lc.point1 - lc.point2).norm();
            if (lc.length >= 0.1f)
                continue;

            lc.plane.reset(new pcl::ModelCoefficients);
            lc.plane->values.push_back(lc.yLocal.x());
            lc.plane->values.push_back(lc.yLocal.y());
            lc.plane->values.push_back(lc.yLocal.z());
            lc.plane->values.push_back(-lc.yLocal.x() * lc.point.x() - lc.yLocal.y() * lc.point.y() - lc.zLocal.z() * lc.point.z());

            if (lc.length > m_maxLineChainLength)
            {
                m_maxLineChainLength = lc.length;
            }

            m_chains.append(lc);
            pcl::PointXYZ pt;
            pt.getVector3fMap() = lc.point;
            middlePointCloud->points.push_back(pt);
            //}
            qDebug().nospace().noquote() << index1 << "-->" << index2 << ": degrees = " << qRadiansToDegrees(radians);// << ", valid = " << valid;
        }
    }

    if (m_chains.isEmpty())
        return;

    qDebug() << "Sorting line chains.";
    qSort(m_chains.begin(), m_chains.end(), [=](const LineChain& lc1, const LineChain& lc2) -> bool
        {
            return qAbs(lc1.radians - M_PI / 2) < qAbs(lc2.radians - M_PI / 2);
        }
    );

    for (int i = 0; i < m_chains.size(); i++)
    {
        LineChain& lc = m_chains[i];
        qDebug().nospace().noquote() << lc.line1 << "-->" << lc.line2 << ": degrees = " << qRadiansToDegrees(lc.radians);
    }

    /*pcl::PCA<pcl::PointXYZ> pca;
    pca.setInputCloud(middlePointCloud);
    Eigen::Vector3f eigenValues = pca.getEigenValues();
    Eigen::Vector3f::Index maxIndex, minIndex;
    eigenValues.maxCoeff(&maxIndex);
    eigenValues.minCoeff(&minIndex);
    QList<int> dims = { 0, 1, 2 };
    dims.removeOne(minIndex);
    dims.removeOne(maxIndex);

    m_lcLocalMiddle = pca.getMean().head(3);
    Eigen::Vector3f rot;
    rot.col(0) = pca.getEigenVectors().col(maxIndex).normalized();
    rot.col(1) = pca.getEigenVectors().col(dims[0]).normalized();
    rot.col(2) = pca.getEigenVectors().col(minIndex).normalized();

    m_lcLocalTransform = Eigen::Matrix4f::Identity();
    m_lcLocalTransform.topLeftCorner(3, 3) = pca.getEigenVectors();
    m_lcLocalTransform.col(3) = pca.getMean();*/
}

void LineExtractor::generateDescriptors()
{
    if (m_chains.size() < 2)
        return;

    m_descriptors.reset(new pcl::PointCloud<LineDescriptor>);
    float distTick = m_boundBoxDiameter / 2 / (LINE_MATCHER_DIVISION - 1);
    qDebug() << "distTick =" << distTick;
    for (int i = 0; i < m_chains.size(); i++)
    {
        LineChain& lc1 = m_chains[i];
        LineDescriptor descriptor;
        QString out;
        for (int j = 0; j < m_chains.size(); j++)
        {
            if (j == i)
                continue;

            LineChain& lc2 = m_chains[j];
            Eigen::Vector3f dir = lc2.point - lc1.point;
            Eigen::Vector3f normal = dir.normalized();

            float cosY = normal.dot(lc1.yLocal);
            float radiansY = qAcos(cosY);
            float radiansX = qAcos(normal.cross(lc1.yLocal).dot(lc1.zLocal));
            int x = static_cast<int>(radiansX / (M_PI / (LINE_MATCHER_DIVISION - 1)));
            int y = static_cast<int>(radiansY / (M_PI / (LINE_MATCHER_DIVISION - 1)));
            int z = static_cast<int>(dir.norm() / distTick);
            //int dim = y * LINE_MATCHER_DIVISION + x;
            //int dim2 = LINE_MATCHER_ANGLE_ELEMDIMS + static_cast<int>(dir.norm() / distTick);
            //descriptor.elems[dim] = descriptor.elems[dim] + 1;
            //descriptor.elems[dim2] = descriptor.elems[dim2] + 2; 
            //out.append(QString("[%1, %2, %3]").arg(radiansX).arg(radiansY).arg(dir.norm()));
            descriptor.elems[x * LINE_MATCHER_DIVISION * LINE_MATCHER_DIVISION + y * LINE_MATCHER_DIVISION + z] += 1;
        }
        QString line;
        for (int i = 0; i < LineDescriptor::elemsSize(); i++)
        {
            line.append(QString::number(descriptor.elems[i]));
            line.append(" ");
        }
        qDebug().noquote() << i << "." << lc1.name() << ":" << line;
        //qDebug().noquote() << i << "." << lc1.name() << ":" << out;
        m_descriptors->points.push_back(descriptor);
    }
}

void LineExtractor::generateDescriptors2()
{
    if (m_chains.size() < 2)
        return;

    float distTick = m_boundBoxDiameter / 2 / (LINE_MATCHER_DIVISION - 1);

    m_descriptors2.reset(new pcl::PointCloud<LineDescriptor2>);
    for (int i = 0; i < m_chains.size(); i++)
    {
        LineDescriptor2 descriptor;
        LineChain& lc = m_chains[i];

        Eigen::Matrix3f localMat;
        localMat.col(0) = lc.xLocal;
        localMat.col(1) = lc.yLocal;
        localMat.col(2) = lc.zLocal;

        ////Eigen::Vector3f dir(Eigen::Vector3f::Zero());
        ////Eigen::Vector3f normal(Eigen::Vector3f::Zero());
        //for (int j = 0; j < m_chains.size(); j++)
        //{
        //    if (j == i)
        //        continue;

        //    LineChain& lc2 = m_chains[j];

        //    Eigen::Vector3f dir = lc2.point - lc.point;
        //    //dir = localMat * dir;
        //    //Eigen::Vector3f normal = dir.normalized();

        //    int x = static_cast<int>(dir.x() / distTick);
        //    int y = static_cast<int>(dir.y() / distTick);
        //    int z = static_cast<int>(dir.z() / distTick);
        //    //descriptor.elems[x * LINE_MATCHER_DIVISION * LINE_MATCHER_DIVISION + y * LINE_MATCHER_DIVISION + z + 5] += 1;

        //    //dir += (lc2.point - lc.point);
        //    //normal += localMat * lc2.yLocal;
        //}
        //dir /= (m_chains.size() - 1);
        //normal /= (m_chains.size() - 1);

        //qDebug() << m_lcLocalMiddle;
        //Eigen::Vector3f lcPos = (lc.point - m_lcLocalMiddle) / m_boundBoxDiameter;
        //qDebug() << lcPos;

        descriptor.elems[0] = lc.radians;// / (M_PI * 2);
        descriptor.elems[1] = lc.length;
        descriptor.elems[2] = lc.point.x();
        descriptor.elems[3] = lc.point.y();
        descriptor.elems[4] = lc.point.z();
        descriptor.elems[5] = m_mslCloud->points[lc.line1].dir.x();
        descriptor.elems[6] = m_mslCloud->points[lc.line1].dir.y();
        descriptor.elems[7] = m_mslCloud->points[lc.line1].dir.z();
        descriptor.elems[8] = m_mslCloud->points[lc.line2].dir.x();
        descriptor.elems[9] = m_mslCloud->points[lc.line2].dir.y();
        descriptor.elems[10] = m_mslCloud->points[lc.line2].dir.z();
        /*descriptor.elems[2] = dir.x() / m_boundBoxDiameter;
        descriptor.elems[3] = dir.y() / m_boundBoxDiameter;
        descriptor.elems[4] = dir.z() / m_boundBoxDiameter;
        descriptor.elems[5] = normal.x() * 5;
        descriptor.elems[6] = normal.y() * 5;
        descriptor.elems[7] = normal.z() * 5;
        descriptor.elems[8] = lcPos.x();
        descriptor.elems[9] = lcPos.y();
        descriptor.elems[10] = lcPos.z();*/

        QString line;
        for (int i = 0; i < LineDescriptor2::elemsSize(); i++)
        {
            line.append(QString::number(static_cast<double>(descriptor.elems[i]), 'g', 3));
            line.append(" ");
        }

        qDebug().noquote() << i << line;
        m_descriptors2->points.push_back(descriptor);
    }
    m_descriptors2->width = m_descriptors2->points.size();
    m_descriptors2->height = 1;
    m_descriptors2->is_dense = true;
}

