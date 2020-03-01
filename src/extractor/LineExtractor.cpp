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

    //QList<LineSegment> lines;
    m_lineSegments.clear();

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

        //qDebug() << i << "error =" << error << (error < m_searchErrorThreshold);
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

        // ��ǰ����ͶӰ������ԭ���������ֱ�ߵ�ͶӰ�����γɵ�ֱ�ߣ������ֱ������ȷ�����pca������ĽǶ�
        Eigen::Vector3f lineVertProj = (point - pointProjToPrimeDir).normalized();
        float cosZ = lineVertProj.dot(zAxis.cross(primeDir).normalized());
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
    m_lineCloud.reset(new pcl::PointCloud<PointLine>);
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

                int count = 0;
                Eigen::Vector3f cecDir(Eigen::Vector3f::Zero());
                // ʹ��������ŷʽ�ۼ������������������ص���x�������������������yz�����������
                pcl::ConditionalEuclideanClustering<pcl::PointXYZI> cec(true);
                cec.setConditionFunction([=](const pcl::PointXYZI& ptSeed, const pcl::PointXYZI& ptCandidate, float sqrDistance) -> bool
                    {
                        if (qSqrt(sqrDistance) < m_mappingTolerance)
                        {
                            Eigen::Vector3f seedPoint = ptSeed.getVector3fMap();
                            Eigen::Vector3f candidatePoint = ptCandidate.getVector3fMap();

                            Eigen::Vector3f dir = m_dirCloud->points[index].getVector3fMap();
                            float distToZ = (candidatePoint - seedPoint).cross(dir).norm();
                            if (distToZ < m_regionGrowingZDistanceThreshold)
                            {
                                return true;
                            }
                        }
                        return false;
                    }
                );
                cec.setClusterTolerance(m_mappingTolerance);
                cec.setMinClusterSize(m_angleMinNeighbours);
                cec.setMaxClusterSize(subCloudIndices.size());
                cec.setSearchMethod(tree);
                cec.setInputCloud(m_mappingCloud);
                cec.setIndices(pcl::IndicesPtr(new std::vector<int>(subCloudIndices)));
                cec.segment(clusterIndices);

                qDebug() << "  cluster size:" << clusterIndices.size();
                std::vector<pcl::PointIndices>::iterator itClusters = clusterIndices.begin();
                while (itClusters != clusterIndices.end())
                {
                    // ���ָ���ľۼ���ȡ��ֱ��
                    pcl::PointIndices indices = *itClusters;
                    Eigen::Vector3f dir(0, 0, 0);
                    Eigen::Vector3f start(0, 0, 0);
                    Eigen::Vector3f end(0, 0, 0);
                    Eigen::Vector3f center(0, 0, 0);

                    // �ж���ɾۼ��ĵ����Ƿ�����Ҫ��
                    if (indices.indices.size() >= m_angleMinNeighbours)
                    {
                        // ����ȫ����ڵ㷽���ƽ���ķ�����Ϊ������ͬʱ�����ʵ㡣
                        for (std::vector<int>::const_iterator itIndices = indices.indices.begin(); itIndices != indices.indices.end(); ++itIndices)
                        {
                            // ȡ��ÿһ���������ֵ
                            int localIndex = m_mappingCloud->points[*itIndices].intensity;
                            // ȡ�����Ӧ��ʵ����ά��
                            pcl::PointXYZI ptBoundary = m_boundaryCloud->points[localIndex];
                            // ȡ��֮ǰ���㲢����ĸõ�PCA������������ע��ֻ����һ�����������
                            pcl::PointXYZI ptDir = m_dirCloud->points[localIndex];

                            // ��dir�����ۼ�
                            dir += ptDir.getVector3fMap();
                            // �����������ۼ�
                            center += ptBoundary.getVector3fMap();

                            ptBoundary.intensity = index;
                            m_linedCloud->push_back(ptBoundary);
                        }
                        // ���ֵ�󣬵õ���ǰ����ۼ���ƽ�����������
                        dir /= indices.indices.size();
                        dir.normalized();
                        center /= indices.indices.size();

                        // ��ֱ֤�ߣ�����ÿ���㵽Ŀ��ֱ�ߵľ���������ʹ�ø�ֵ��Ϊ�������֤�Ƿ�Ϊ��Ч��ֱ�ߡ�
                        // ͬʱ������һ����γɵ�������յ㡣
                        float error = 0;
                        float avgError = 0;
                       
                        std::vector<float> errors(indices.indices.size());
                        int count = 0;
                        for (std::vector<int>::const_iterator itIndices = indices.indices.begin(); itIndices != indices.indices.end(); ++itIndices)
                        {
                            int index = m_mappingCloud->points[*itIndices].intensity;
                            pcl::PointXYZI ptBoundary = m_boundaryCloud->points[index];
                            Eigen::Vector3f boundaryPoint = ptBoundary.getVector3fMap();
                            float e = (boundaryPoint - center).cross(dir).norm() * 1000;
                            errors[count++] = e;
                            avgError += e;

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
                        avgError /= indices.indices.size();
                        for (int i = 0; i < errors.size(); i++)
                        {
                            error += qPow(errors[i] - avgError, 2);
                        }
                        error /= errors.size();
                        qDebug() << "   cluster size:" << indices.indices.size() << ", error:" << error;

                        LineSegment ls(start, end);
                        if (ls.length() > m_minLineLength && error <= 200)
                        {
                            m_lineSegments.append(ls);
                            m_errors.append(error);
                            m_linePointsCount.append(indices.indices.size());

                            PointLine pl;
                            Eigen::Vector3f dir = ls.direction().normalized();
                            calculateAlphaBeta(dir, pl.dAngleX, pl.dAngleY);
                            pl.dist = start.cross(dir).norm();
                            calculateAlphaBeta((start.normalized() - dir).normalized(), pl.vAngleX, pl.vAngleY);
                            m_lineCloud->points.push_back(pl);
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

    
    qDebug() << "[LineExtractor::compute] exit";
    return m_lineSegments;
}

void LineExtractor::extractLinesFromPlanes(const QList<Plane>& planes)
{
    //m_mslPointCloud.reset(new pcl::PointCloud<MSLPoint>);
    //m_mslCloud.reset(new pcl::PointCloud<Line>);
    if (planes.isEmpty())
        return;

    Eigen::Vector3f globalDir(1, 1, 1);

    if (!m_mslCloud)
    {
        m_mslCloud.reset(new pcl::PointCloud<Line>);
    }
    if (!m_lineCloud)
    {
        m_lineCloud.reset(new pcl::PointCloud<PointLine>);
    }

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
            //MSLPoint mslPoint;
            //float alpha, beta;
            //calculateAlphaBeta(crossLine, alpha, beta);
            //mslPoint.alpha = alpha / M_PI;
            //mslPoint.beta = beta / M_PI;
            //mslPoint.x = closedPoint.x() / m_boundBoxDiameter;
            //mslPoint.y = closedPoint.y() / m_boundBoxDiameter;
            //mslPoint.z = closedPoint.z() / m_boundBoxDiameter;
            //m_mslPointCloud->push_back(mslPoint);
            Eigen::Vector3f start = closedPoint - crossLine * 0.075f;
            Eigen::Vector3f end = closedPoint + crossLine * 0.075f;
            LineSegment ls(start, end);
            m_lineSegments.append(ls);

            PointLine pl;
            Eigen::Vector3f dir = ls.direction().normalized();
            calculateAlphaBeta(dir, pl.dAngleX, pl.dAngleY);
            pl.dist = start.cross(dir).norm();
            calculateAlphaBeta((start.normalized() - dir).normalized(), pl.vAngleX, pl.vAngleY);
            m_lineCloud->points.push_back(pl);

            /*Line msl;
            msl.dir = crossLine;
            msl.point = closedPoint;
            msl.weight = 1;
            m_mslCloud->push_back(msl);*/
        }
    }

    //m_mslPointCloud->width = m_mslPointCloud->points.size();
    //m_mslPointCloud->height = 1;
    //m_mslPointCloud->is_dense = true;
}

void LineExtractor::segmentLines()
{
    m_mslCloud.reset(new pcl::PointCloud<Line>);
    if (m_lineSegments.size() > 0)
    {
        pcl::KdTreeFLANN<PointLine> lineTree;
        lineTree.setInputCloud(m_lineCloud);

        QMap<int, bool> lineProcessed;
        QVector<float> lineDense(m_lineSegments.size());
        QVector<int> lineIndices(m_lineSegments.size());
        float radius = M_PI / 36;
        for (int i = 0; i < m_lineSegments.size(); i++)
        {
            std::vector<int> indices;
            std::vector<float> dists;
            lineTree.radiusSearch(m_lineCloud->points[i], radius, indices, dists);
            lineDense[i] = indices.size();
            lineIndices[i] = i;

            qDebug() << "line" << i << ", dense:" << indices.size();
        }

        qSort(lineIndices.begin(), lineIndices.end(), [=](int a, int b) -> bool
            {
                return lineDense[a] > lineDense[b];
            }
        );

        for (int i = 0; i < lineIndices.size(); i++)
        {
            int index = lineIndices[i];
            if (lineProcessed.contains(index))
            {
                continue;
            }

            PointLine pl = m_lineCloud->points[index];
            std::vector<int> indices;
            std::vector<float> dists;
            lineTree.radiusSearch(m_lineCloud->points[index], radius, indices, dists);
            qDebug() << "line" << index << ", merging lines:" << indices.size();
            Eigen::Vector3f dir(Eigen::Vector3f::Zero());
            Eigen::Vector3f point(Eigen::Vector3f::Zero());
            float sumLength = 0;
            for (int i = 0; i < indices.size(); i++)
            {
                int neighbourIndex = indices[i];
                qDebug() << "    " << neighbourIndex;
                lineProcessed[neighbourIndex] = true;
                LineSegment ls = m_lineSegments[neighbourIndex];
                sumLength += ls.length();
            }
            for (int i = 0; i < indices.size(); i++)
            {
                int neighbourIndex = indices[i];
                LineSegment ls = m_lineSegments[neighbourIndex];
                dir += ls.direction().normalized() * ls.length() / sumLength;
                point += ls.middle() * ls.length() / sumLength;
            }

            LineSegment ls = m_lineSegments[index];
            Line line;
            line.dir = dir;
            line.point = point + dir * (point.dot(dir));
            line.weight = 1;
            m_mslCloud->push_back(line);
        }
        m_mslCloud->width = m_mslCloud->points.size();
        m_mslCloud->height = 1;
        m_mslCloud->is_dense = true;
    }
}

void LineExtractor::generateLineChains()
{
    m_chains.clear();

    //// ��������ֱ�߷����Э�������
    //Eigen::VectorXf meanDir;
    //Eigen::MatrixXf mat;
    //mat.resize(3, m_mslCloud->size());
    //for (int i = 0; i < m_mslCloud->size(); i++)
    //{
    //    mat.col(i) = m_mslCloud->points[i].dir;
    //    //avgDir += m_mslCloud->points[i].dir;
    //}
    //meanDir = mat.rowwise().mean();
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
    qDebug() << "[LineExtractor::generateLineChains] msl cloud size:";
    qDebug() << m_mslCloud->size();
    if (m_mslCloud->size() == 0)
        return;

    for (int i = 0; i < m_mslCloud->size(); i++)
    {
        for (int j = i + 1; j < m_mslCloud->size(); j++)
        {
            float radians = 0;
            QString group = QString("%1_%2").arg(i).arg(j);
            //qDebug() << "generating line chain:" << group;
            {
                LineChain lc;
                
                lc.group = group;
                lc.lineNo1 = i;
                lc.lineNo2 = j;
                lc.line1 = m_mslCloud->points[i];
                lc.line2 = m_mslCloud->points[j];

                // ������ֱ�߽Ƕȣ�����ѡ�ýǶȵľ���ֵ
                radians = qAbs(qAcos(lc.line1.dir.dot(lc.line2.dir)));
                // ���С����ֵ��������
                if (radians < M_PI / 4)
                    continue;
                // ����������㣬������û���ֵ
                lc.radians = radians;

                if (!generateLineChain(lc))
                    continue;
            }

            {
                LineChain lc;
                lc.group = group;
                lc.radians = radians;
                lc.lineNo1 = j;
                lc.lineNo2 = i;
                lc.line1 = m_mslCloud->points[j];
                lc.line2 = m_mslCloud->points[i];
                generateLineChain(lc);
            }
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
        qDebug().nospace().noquote() << lc.lineNo1 << "-->" << lc.lineNo2 << ": degrees = " << qRadiansToDegrees(lc.radians);
    }
}

bool LineExtractor::generateLineChain(LineChain& lc)
{
    // �����ֲ�����ϵ
    lc.xLocal = lc.line1.dir;   // ��һֱ�ߵķ�����Ϊx��
    lc.yLocal = lc.line1.dir.cross(lc.line2.dir).normalized();  // ��˷�����Ϊy��
    lc.zLocal = lc.xLocal.cross(lc.yLocal).normalized();        // �ٲ�˷�����Ϊz��

    // ���������
    Eigen::Vector3f p1;     // ��ʾֱ��1�ϵ�һ����
    Eigen::Vector3f p2;     // ��ʾֱ��2�ϵ�һ����
    Eigen::Vector3f c1;     // ��ʾ����ֱ�ߵĲ��ֱ�߷���P1-P2
    Eigen::Vector3f c2;     // ��ʾ����ֱ�ߵĲ��ֱ�߷���P2-P1
    Eigen::Vector3f d1;     // ��ʾ��1�ķ���
    Eigen::Vector3f d2;     // ��ʾ��2�ķ���
    float t1;               // P1��cross1�ľ���
    float t2;               // P2��cross2�ľ���
    Eigen::Vector3f cross1; // ��1�ϵ������
    Eigen::Vector3f cross2; // ��2�ϵ������
    float l;                // ����ֱ�߼�Ĵ��߾���

    p1 = lc.line1.point;
    p2 = lc.line2.point;
    d1 = lc.line1.dir.normalized();
    d2 = lc.line2.dir.normalized();
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
    {
        return false;
    };

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
    qDebug().nospace().noquote() << lc.lineNo1 << "-->" << lc.lineNo2 << ": degrees = " << qRadiansToDegrees(lc.radians);
    return true;
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

        descriptor.elems[0] = lc.radians;       // ��˻���ֵ
        descriptor.elems[1] = lc.length;        // ��̴��߶γ���
        descriptor.elems[2] = lc.point.x();     // �е��������
        descriptor.elems[3] = lc.point.y();
        descriptor.elems[4] = lc.point.z();
        descriptor.elems[5] = lc.line1.dir.x(); // ��1�����������
        descriptor.elems[6] = lc.line1.dir.y();
        descriptor.elems[7] = lc.line1.dir.z();
        descriptor.elems[8] = lc.line2.dir.x(); // ��2�����������
        descriptor.elems[9] = lc.line2.dir.y();
        descriptor.elems[10] = lc.line2.dir.z();
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

