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

    // ���㵱ǰ��������е����ֱ������ֵ�����ڶ�ֱ�߳��Ƚ��й�һ���ķ�ĸ��
    Eigen::Vector4f minPoint, maxPoint;
    pcl::getMinMax3D<pcl::PointXYZI>(*boundaryCloud, minPoint, maxPoint);
    m_minPoint = minPoint.head(3);
    m_maxPoint = maxPoint.head(3);
    m_boundBoxDiameter = (m_maxPoint - m_minPoint).norm();
    qDebug() << "bound box diameter:" << m_boundBoxDiameter << minPoint.x() << minPoint.y() << minPoint.z() << maxPoint.x() << maxPoint.y() << maxPoint.z() << (m_maxPoint - m_minPoint).norm();

    pcl::search::KdTree<pcl::PointXYZI> tree;
    tree.setInputCloud(boundaryCloud);

    // globalDir����ͳһ���е�ֱ�߷���
    Eigen::Vector3f globalDir(1, 1, 1);
    globalDir.normalize();
    Eigen::Vector3f xAxis(1, 0, 0);
    Eigen::Vector3f yAxis(0, 1, 0);
    Eigen::Vector3f zAxis(0, 0, 1);

    // ����ÿһ���߽���PCA������
    int index = 0;
    for (int i = 0; i < boundaryCloud->size(); i++)
    {
        pcl::PointXYZI ptIn = boundaryCloud->points[i];
        Eigen::Vector3f point = ptIn.getVector3fMap();

        // ���ҽ���
        std::vector<int> neighbourIndices;
        std::vector<float> neighbourDistances;
        tree.radiusSearch(ptIn, m_searchRadius, neighbourIndices, neighbourDistances);

        // ���ڵ�̫�ٱ�ʾ����һ����Ⱥ�㣬��С��3Ҳ�޷�����PCA����
        if (neighbourIndices.size() < m_minNeighbours)
        {
            continue;
        }

        // PCA���㵱ǰ���������
        pcl::PCA<pcl::PointXYZI> pca;
        pca.setInputCloud(boundaryCloud);
        pca.setIndices(pcl::IndicesPtr(new std::vector<int>(neighbourIndices)));
        Eigen::Vector3f eigenValues = pca.getEigenValues();
        Eigen::Vector3f::Index maxIndex;
        eigenValues.maxCoeff(&maxIndex);

        // ������
        Eigen::Vector3f primeDir = pca.getEigenVectors().col(maxIndex).normalized();
        // �е�
        Eigen::Vector3f meanPoint = pca.getMean().head(3);

        // �������
        float error = 0;
        for (int j = 0; j < neighbourIndices.size(); j++)
        {
            int neighbourIndex = neighbourIndices[j];
            pcl::PointXYZI ptNeighbour = boundaryCloud->points[neighbourIndex];
            // ��ֱ����
            float vertDist = primeDir.cross(ptNeighbour.getVector3fMap() - meanPoint).norm();
            error += vertDist;
        }
        error /= neighbourIndices.size();

        // ������ֵ̫��������
        if (error >= m_searchErrorThreshold)
        {
            continue;
        }

        // ��֤��ȫ�ַ���ͬ��
        if (primeDir.dot(globalDir) < 0)
        {
            primeDir = -primeDir;
        }

        Eigen::Vector3f eulerAngles(Eigen::Vector3f::Zero());
        // ʹ�ö�ά����ά��ӳ��ǡ���άʹ�ü�����ĸ����Ǻͺ����
        if (m_angleMappingMethod == TWO_DIMS)
        {
            float alpha, beta;
            // ���㸩���Ǻ�ˮƽ��
            calculateAlphaBeta(primeDir, alpha, beta);

            eulerAngles.x() = alpha * M_1_PI;
            eulerAngles.y() = beta * M_1_PI;
        }
        else if (m_angleMappingMethod = THREE_DIMS)
        {
            // ����ŷ����
            Eigen::Quaternionf rotQuat = Eigen::Quaternionf();
            rotQuat.setFromTwoVectors(globalDir, primeDir);
            Eigen::Matrix3f rotMatrix = rotQuat.toRotationMatrix();
            eulerAngles = rotMatrix.eulerAngles(0, 1, 2) * M_1_PI;
        }

        // ���㵱ǰ�㵽����ԭ�������������ֱ�ߵĴ�ֱ����
        float distance = point.cross(primeDir).norm();

        // ���㵱ǰ��ͶӰ������ԭ�������������ֱ�ߵ�ͶӰ������򳤶�
        Eigen::Vector3f pointProjToPrimeDir = primeDir * point.dot(primeDir);
        float xCoord = pointProjToPrimeDir.norm();
        if (pointProjToPrimeDir.dot(primeDir) < 0)
        {
            xCoord = -xCoord;
        }

        // ��ǰ����ͶӰ������ԭ���������ֱ�ߵ�ͶӰ�����γɵ�ֱ�ߣ������ֱ������ȷ���ĽǶ�
        Eigen::Vector3f lineVertProj = (point - pointProjToPrimeDir).normalized();
        float cosZ = lineVertProj.dot(zAxis);
        float radiansZ = qAcos(cosZ);
        if (lineVertProj.cross(zAxis).dot(primeDir) < 0)
        {
            radiansZ = (M_PI * 2) - radiansZ;
        }

        // ������������Ƕ�ֵ����
        pcl::PointXYZI anglePt;
        anglePt.x = eulerAngles.x();
        anglePt.y = eulerAngles.y();
        anglePt.z = eulerAngles.z();
        anglePt.intensity = index;
        m_angleCloud->push_back(anglePt);
        
        // ������������ƣ���Ҫ��Ϊ�˱��������򣬹���������ʹ�á�
        pcl::PointXYZI dirPt;
        dirPt.getVector3fMap() = primeDir;
        dirPt.intensity = index;
        m_dirCloud->push_back(dirPt);

        // ��������ԭ�㴹�ࡢ��������нǺ�������ά���������������γ�һ���µ�ӳ����ƣ�����ʹ�þۼ���ȡ������ֱ�ߡ�
        pcl::PointXYZI mappingPt;
        mappingPt.x = distance / m_boundBoxDiameter * 2;
        mappingPt.y = radiansZ / (M_PI * 2);
        mappingPt.z = xCoord / m_boundBoxDiameter * 2;
        mappingPt.intensity = index;
        m_mappingCloud->push_back(mappingPt);

        // ������������ĵ��������һ�����ƣ�����������㡣
        pcl::PointXYZI centerPt;
        centerPt.getVector3fMap() = meanPoint;
        centerPt.intensity = index;
        m_centerCloud->push_back(centerPt);

        // ��������ĵ����ܶȵ��������һ�����ƣ�����������㡣
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

    // �������������ƵĲ���������������ͨ���Ƕ�ֵ������������������ͬ�ľۼ�
    pcl::search::KdTree<pcl::PointXYZI> angleCloudTree;
    angleCloudTree.setInputCloud(m_angleCloud);

    // �������ѭ�����ڲ�����������ͬ�ĵ㣬������ʹ���������������������ϴ��ƫ��
    //float maxDensity = 0;
    //float minDensity = m_angleCloud->size();
    QMap<int, std::vector<int>> neighbours;
    for (int i = 0; i < m_angleCloud->size(); i++)
    {
        pcl::PointXYZI ptAngle = m_angleCloud->points[i];

        // ���ҽ���
        std::vector<int> neighbourIndices;
        std::vector<float> neighbourDistances;
        angleCloudTree.radiusSearch(ptAngle, m_angleSearchRadius, neighbourIndices, neighbourDistances);
        neighbours.insert(i, neighbourIndices);
        
        // �����ܶ�ֵ
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

        // ��������ֵ
        Eigen::Vector3f center(Eigen::Vector3f::Zero());
        for (int n = 0; n < neighbourIndices.size(); n++)
        {
            int neighbourIndex = neighbourIndices[n];
            pcl::PointXYZI ptNeighbour = m_angleCloud->points[neighbourIndex];
            center += ptNeighbour.getVector3fMap();
        }
        center /= density;

        // ����������
        float offsetRate = (ptAngle.getVector3fMap() - center).norm();
        m_offsetRate.append(offsetRate);
    }

    // �����ܶ�ֵ��������ֵ�����е�ӳ����������
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

    // ���ܶ�����ֵ��ʼ����ÿһ��ӳ��㣬ÿһ��ӳ��㼰����ڵ㼴Ϊһ��ͬ����ĵ㣬����Щ��ķ���
    // ���ܽ�����ƽ�ж�����Զ��
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

            // �����ǰӳ���Ľ��������㹻�����ٴ�ʹ��������ŷʽ�ۼ�����������ٴξۼ������ҳ��������������߶Ρ�
            if (subCloudIndices.size() >= m_angleMinNeighbours)
            {
                m_subCloudIndices.insert(m_angleCloud->points[index].intensity, subCloudIndices);

                qDebug() << "index:" << index << ", sub cloud size:" << subCloudIndices.size();

                pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
                tree->setInputCloud(m_mappingCloud);

                std::vector<pcl::PointIndices> clusterIndices;

                // ʹ��������ŷʽ�ۼ������������������ص���Z�������������������XY�����������
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
                        // ����ȫ����ڵ㷽���ƽ���ķ�����Ϊ������ͬʱ�����ʵ㡣
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

                        // ��ֱ֤�ߣ�����ÿ���㵽Ŀ��ֱ�ߵľ���������ʹ�ø�ֵ��Ϊ�������֤�Ƿ�Ϊ��Ч��ֱ�ߡ�
                        // ͬʱ������һ����γɵ�������յ㡣
                        float error = 0;
                       
                        for (std::vector<int>::const_iterator itIndices = indices.indices.begin(); itIndices != indices.indices.end(); ++itIndices)
                        {
                            int index = m_mappingCloud->points[*itIndices].intensity;
                            pcl::PointXYZI ptBoundary = m_boundaryCloud->points[index];
                            Eigen::Vector3f boundaryPoint = ptBoundary.getVector3fMap();
                            error = (boundaryPoint - center).cross(dir).norm();

                            if (start.isZero())
                            {
                                // �����һ��ѭ�����õ�ǰ����Ϊ���
                                start = closedPointOnLine(boundaryPoint, dir, center);
                            }
                            else
                            {
                                // ����ǰ���뵱ǰ���������ʱ�������һ�𣬲鿴���뵱ǰ�ۼ��������һ���ԣ���һ����ǰ��Ϊ�µ���㡣
                                if ((start - boundaryPoint).dot(dir) > 0)
                                {
                                    start = closedPointOnLine(boundaryPoint, dir, center);
                                }
                            }

                            if (end.isZero())
                            {
                                // �����һ��ѭ�����õ�ǰ����Ϊ�յ�
                                end = closedPointOnLine(boundaryPoint, dir, center);
                            }
                            else
                            {
                                // ����ǰ���뵱ǰ���������ʱ�յ�����һ�𣬲鿴���뵱ǰ�ۼ��������һ���ԣ���һ����ǰ��Ϊ�µ��յ㡣
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

    // �ٴζ��߶ν���ӳ�䣬ÿһ���߶�ӳ��Ϊһ��5ά������
    // ����ǰ����ά��Ϊ�����Ǻͺ���ǣ�������ά����ԭ�㵽ֱ�ߴ��߽����λ�á�
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
            if (qAbs(cos) > 0.95f)
                continue;

            Eigen::Vector3f crossLine = plane1.dir.cross(plane2.dir).normalized();
            if (crossLine.dot(globalDir) < 0)
                crossLine = -crossLine;
            if (crossLine.isZero())
                continue;

            // ���ƽ��1�ϵĵ㵽���ߵĴ��߷���
            Eigen::Vector3f lineOnPlane1 = plane1.dir.cross(crossLine).normalized();
            lineOnPlane1.normalize();
            float dist1 = (plane2.point - plane1.point).dot(plane2.dir) / (lineOnPlane1.dot(plane2.dir));
            // ���ƽ��1�ϵĵ��ڽ�����ͶӰ�Ľ���
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

