#include "DDBPLineExtractor.h"

#include <QtMath>
#include <QDebug>

#include <pcl/common/pca.h>
#include <pcl/search/search.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/segmentation/extract_clusters.h>

DDBPLineExtractor::DDBPLineExtractor(QObject* parent)
    : QObject(parent)
    , m_searchRadius(0.05f)
    , m_minNeighbourCount(3)
    , m_searchErrorThreshold(0.025f)
{

}

QList<LineSegment> DDBPLineExtractor::compute(const pcl::PointCloud<pcl::PointXYZI>::Ptr& boundaryCloud)
{
    m_boundaryCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    m_angleCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    m_dirCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    m_mappingCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    m_centerCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);

    Eigen::Vector4f minPoint, maxPoint;
    pcl::getMinMax3D<pcl::PointXYZI>(*boundaryCloud, minPoint, maxPoint);
    m_minPoint = minPoint.head(3);
    m_maxPoint = maxPoint.head(3);
    m_boundBoxDiameter = (m_maxPoint - m_minPoint).norm();

    qDebug() << "bound box diameter:" << m_boundBoxDiameter << minPoint.x() << minPoint.y() << minPoint.z() << maxPoint.x() << maxPoint.y() << maxPoint.z();

    pcl::search::KdTree<pcl::PointXYZI> tree;
    tree.setInputCloud(boundaryCloud);

    Eigen::Vector3f globalDir(1, 1, 1);
    globalDir.normalize();
    Eigen::Vector3f xAxis(1, 0, 0);
    Eigen::Vector3f yAxis(0, 1, 0);
    Eigen::Vector3f zAxis(0, 0, 1);

    m_boundaryIndices.reset(new std::vector<int>);
    int index = 0;
    for (int i = 0; i < boundaryCloud->size(); i++)
    {
        pcl::PointXYZI ptIn = boundaryCloud->points[i];
        Eigen::Vector3f point = ptIn.getVector3fMap();

        // ���ҽ���
        std::vector<int> neighbourIndices;
        std::vector<float> neighbourDistances;
        tree.radiusSearch(ptIn, m_searchRadius, neighbourIndices, neighbourDistances);

        // ���ڵ�̫��ʾ����һ����Ⱥ�㣬��С��3Ҳ�޷�����PCA����
        if (neighbourIndices.size() < m_minNeighbourCount)
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

        // ���㸩���ǵĲ���
        float cosY = primeDir.dot(yAxis);
        float alpha = qAcos(cosY);

        // ����ˮƽ��
        float cosX = (primeDir - yAxis * cosY).normalized().dot(xAxis);
        float beta = qAcos(cosX);

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
            radiansZ = M_2_PI - radiansZ;
        }

        pcl::PointXYZI anglePt;
        anglePt.x = alpha / M_PI;
        anglePt.y = beta / M_2_PI;
        anglePt.z = 0;
        anglePt.intensity = index;
        m_angleCloud->push_back(anglePt);
        
        pcl::PointXYZI dirPt;
        dirPt.getVector3fMap() = primeDir;
        dirPt.intensity = index;
        m_dirCloud->push_back(dirPt);

        pcl::PointXYZI mappingPt;
        mappingPt.x = distance / m_boundBoxDiameter / 2;
        mappingPt.y = radiansZ / M_2_PI;
        mappingPt.z = xCoord / m_boundBoxDiameter / 2;
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
        index++;
    }
    m_boundaryCloud->width = m_boundaryCloud->points.size();
    m_boundaryCloud->height = 1;
    m_boundaryCloud->is_dense = true;

    qDebug() << "Input cloud size:" << boundaryCloud->size() << ", filtered cloud size:" << m_boundaryCloud->size();

    return QList<LineSegment>();
}
