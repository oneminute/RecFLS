#include "DDBPLineExtractor.h"

#include <QtMath>
#include <QDebug>
#include <QVector>

#include <pcl/common/pca.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>

#include "util/Utils.h"

DDBPLineExtractor::DDBPLineExtractor(QObject* parent)
    : QObject(parent)
    , m_angleMappingMethod(TWO_DIMS)
    , m_searchRadius(0.05f)
    , m_minNeighbours(3)
    , m_searchErrorThreshold(0.025f)
    , m_angleSearchRadius(qDegreesToRadians(20.0) * M_1_PI)
    , m_angleMinNeighbours(10)
    , m_mappingTolerance(0.01f)
    , m_regionGrowingYDistanceThreshold(0.005f)
    , m_minLineLength(0.01f)
{

}

QList<LineSegment> DDBPLineExtractor::compute(const pcl::PointCloud<pcl::PointXYZI>::Ptr& boundaryCloud)
{
    qDebug() << "angleMappingMethod" << m_angleMappingMethod;
    qDebug() << "searchRadius:" << m_searchRadius;
    qDebug() << "minNeighbours:" << m_minNeighbours;
    qDebug() << "searchErrorThreshold:" << m_searchErrorThreshold;
    qDebug() << "angleSearchRadius:" << m_angleSearchRadius;
    qDebug() << "angleMinNeighbours:" << m_angleMinNeighbours;
    qDebug() << "mappingTolerance:" << m_mappingTolerance;
    qDebug() << "regionGrowingYDistance:" << m_regionGrowingYDistanceThreshold;
    qDebug() << "minLineLength:" << m_minLineLength;

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

    //Eigen::Vector4f minPoint, maxPoint;
    //pcl::getMinMax3D<pcl::PointXYZI>(*boundaryCloud, minPoint, maxPoint);
    //m_minPoint = minPoint.head(3);
    //m_maxPoint = maxPoint.head(3);
    //m_boundBoxDiameter = (m_maxPoint - m_minPoint).norm();
    m_boundBoxDiameter = 4;

    //qDebug() << "bound box diameter:" << m_boundBoxDiameter << minPoint.x() << minPoint.y() << minPoint.z() << maxPoint.x() << maxPoint.y() << maxPoint.z();

    pcl::search::KdTree<pcl::PointXYZI> tree;
    tree.setInputCloud(boundaryCloud);

    Eigen::Vector3f globalDir(1, 1, 1);
    globalDir.normalize();
    Eigen::Vector3f xAxis(1, 0, 0);
    Eigen::Vector3f yAxis(0, 1, 0);
    Eigen::Vector3f zAxis(0, 0, 1);

    int index = 0;
    for (int i = 0; i < boundaryCloud->size(); i++)
    {
        pcl::PointXYZI ptIn = boundaryCloud->points[i];
        Eigen::Vector3f point = ptIn.getVector3fMap();

        // 查找近邻
        std::vector<int> neighbourIndices;
        std::vector<float> neighbourDistances;
        tree.radiusSearch(ptIn, m_searchRadius, neighbourIndices, neighbourDistances);

        // 近邻点太表示这是一个离群点，且小于3也无法进行PCA计算
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
            // 计算俯仰角的补角
            float cosY = primeDir.dot(yAxis);
            float alpha = qAcos(cosY);

            // 计算水平角
            float cosX = (primeDir - yAxis * cosY).normalized().dot(xAxis);
            float beta = qAcos(cosX);

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

        pcl::PointXYZI anglePt;
        anglePt.x = eulerAngles.x();
        anglePt.y = eulerAngles.y();
        anglePt.z = eulerAngles.z();
        anglePt.intensity = index;
        m_angleCloud->push_back(anglePt);
        
        pcl::PointXYZI dirPt;
        dirPt.getVector3fMap() = primeDir;
        dirPt.intensity = index;
        m_dirCloud->push_back(dirPt);

        pcl::PointXYZI mappingPt;
        mappingPt.x = distance / m_boundBoxDiameter * 2;
        mappingPt.y = radiansZ * M_1_PI / 2;
        mappingPt.z = xCoord / m_boundBoxDiameter * 2;
        mappingPt.intensity = index;
        m_mappingCloud->push_back(mappingPt);

        pcl::PointXYZI centerPt;
        centerPt.getVector3fMap() = meanPoint;
        centerPt.intensity = index;
        m_centerCloud->push_back(centerPt);

        pcl::PointXYZI pt = ptIn;
        pt.intensity = 0;
        m_boundaryIndices->push_back(index);
        m_boundaryCloud->push_back(ptIn);

        if (index % 100 == 0)
        {
            //qDebug() << alpha << "(" << qRadiansToDegrees(alpha) << ")," << beta << "(" << qRadiansToDegrees(beta) << ")";
            qDebug() << eulerAngles.x() << eulerAngles.y() << eulerAngles.z();
        }
        index++;
        m_angleCloudIndices.append(index);
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

    // 创建参数化点云的查找树
    pcl::search::KdTree<pcl::PointXYZI> angleCloudTree;
    tree.setInputCloud(m_angleCloud);

    float maxDensity = 0;
    float minDensity = m_angleCloud->size();
    QMap<int, std::vector<int>> neighbours;
    for (int i = 0; i < m_angleCloud->size(); i++)
    {
        pcl::PointXYZI ptAngle = m_angleCloud->points[i];

        // 查找近邻
        std::vector<int> neighbourIndices;
        std::vector<float> neighbourDistances;
        tree.radiusSearch(ptAngle, m_angleSearchRadius, neighbourIndices, neighbourDistances);
        neighbours.insert(i, neighbourIndices);
        
        // 计算密度值
        float density = neighbourIndices.size();

        if (density > maxDensity)
        {
            maxDensity = density;
        }
        if (density < minDensity)
        {
            minDensity = density;
        }

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

    QVector<bool> processed(m_angleCloud->size(), false);
    for (int i = 0; i < m_density.size(); i++)
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
            if (subCloudIndices.size() >= m_angleMinNeighbours)
            {
                m_subCloudIndices.insert(m_angleCloud->points[index].intensity, subCloudIndices);

                qDebug() << "index:" << index << ", sub cloud size:" << subCloudIndices.size();

                pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
                tree->setInputCloud(m_mappingCloud);

                std::vector<pcl::PointIndices> clusterIndices;

                /*pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
                ec.setClusterTolerance(m_mappingTolerance);
                ec.setMinClusterSize(1);
                ec.setMaxClusterSize(subCloudIndices.size());
                ec.setSearchMethod(tree);
                ec.setInputCloud(m_mappingCloud);
                ec.setIndices(pcl::IndicesPtr(new std::vector<int>(subCloudIndices)));
                ec.extract(clusterIndices);*/

                pcl::ConditionalEuclideanClustering<pcl::PointXYZI> cec(true);
                cec.setConditionFunction([=](const pcl::PointXYZI& ptSeed, const pcl::PointXYZI& ptCandidate, float sqrDistance) -> bool
                    {
                        if (qSqrt(sqrDistance) < m_mappingTolerance)
                        {
                            Eigen::Vector3f seedPoint = ptSeed.getVector3fMap();
                            Eigen::Vector3f candidatePoint = ptCandidate.getVector3fMap();

                            Eigen::Vector3f yAxis(0, 1, 0);
                            float distToY = (candidatePoint - seedPoint).cross(yAxis).norm();
                            if (distToY < m_regionGrowingYDistanceThreshold)
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

                        // 验证直线，计算每个点到目标直线的距离期望
                        float error = 0;
                       
                        for (std::vector<int>::const_iterator itIndices = indices.indices.begin(); itIndices != indices.indices.end(); ++itIndices)
                        {
                            int index = m_mappingCloud->points[*itIndices].intensity;
                            pcl::PointXYZI ptBoundary = m_boundaryCloud->points[index];
                            //pcl::PointXYZI ptCenter = m_centerCloud->points[index];
                            Eigen::Vector3f boundaryPoint = ptBoundary.getVector3fMap();
                            //Eigen::Vector3f centerPoint = ptCenter.getVector3fMap();
                            error = (boundaryPoint - center).cross(dir).norm();

                            if (start.isZero())
                            {
                                start = closedPointOnLine(boundaryPoint, dir, center);
                            }
                            else
                            {
                                if ((start - boundaryPoint).dot(dir) > 0)
                                {
                                    start = closedPointOnLine(boundaryPoint, dir, center);
                                }
                            }

                            if (end.isZero())
                            {
                                end = closedPointOnLine(boundaryPoint, dir, center);
                            }
                            else
                            {
                                if ((boundaryPoint - end).dot(dir) > 0)
                                {
                                    end = closedPointOnLine(boundaryPoint, dir, center);
                                }
                            }
                        }
                        error /= indices.indices.size();
                        qDebug() << "    error:" << error;

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

    return lines;
}

