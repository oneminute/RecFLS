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
    //, m_angleMappingMethod(TWO_DIMS)
    //, m_searchRadius(0.05f)
    //, m_minNeighbours(3)
    //, m_searchErrorThreshold(0.025f)
    //, m_angleSearchRadius(qDegreesToRadians(20.0) * M_1_PI)
    //, m_angleMinNeighbours(10)
    //, m_mappingTolerance(0.01f)
    //, m_regionGrowingZDistanceThreshold(0.005f)
    //, m_minLineLength(0.01f)
    //, m_mslRadiusSearch(0.01f)
    //, m_boundBoxDiameter(2)
    //, m_groupLinesRadius(M_PI / 90)
    , PROPERTY_INIT(BoundaryCloudA1dThreshold, 0.4f)
    , PROPERTY_INIT(CornerCloudA1dThreshold, 0.3f)
    , PROPERTY_INIT(BoundaryCloudSearchRadius, 0.1f)
    , PROPERTY_INIT(CornerCloudSearchRadius, 0.2f)
    , PROPERTY_INIT(PCASearchRadius, 0.1f)
    , PROPERTY_INIT(MinNeighboursCount, 10)
    , PROPERTY_INIT(AngleCloudSearchRadius, 20)
    , PROPERTY_INIT(AngleCloudMinNeighboursCount, 10)
    , PROPERTY_INIT(MinLineLength, 0.1f)
    , PROPERTY_INIT(BoundaryLineInterval, 0.1f)
    , PROPERTY_INIT(CornerLineInterval, 0.2f)
    , PROPERTY_INIT(BoundaryMaxZDistance, 0.01f)
    , PROPERTY_INIT(CornerMaxZDistance, 0.05f)
    , PROPERTY_INIT(CornerGroupLinesSearchRadius, 0.05f)
{

}

QList<LineSegment> LineExtractor::compute(const pcl::PointCloud<pcl::PointXYZI>::Ptr& boundaryCloud, const pcl::PointCloud<pcl::PointXYZI>::Ptr& cornerCloud)
{
    qDebug() << "angleMappingMethod" << m_angleMappingMethod;
    //qDebug() << "searchRadius:" << m_searchRadius;
    //qDebug() << "minNeighbours:" << m_minNeighbours;
    //qDebug() << "searchErrorThreshold:" << m_searchErrorThreshold;
    //qDebug() << "angleSearchRadius:" << m_angleSearchRadius;
    //qDebug() << "angleMinNeighbours:" << m_angleMinNeighbours;
    //qDebug() << "mappingTolerance:" << m_mappingTolerance;
    //qDebug() << "z distance threshold:" << m_regionGrowingZDistanceThreshold;
    //qDebug() << "minLineLength:" << m_minLineLength;
    //qDebug() << "msl radius search:" << m_mslRadiusSearch;

    m_cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    m_lineCloud.reset(new pcl::PointCloud<Line>);
    //m_filteredCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    //m_angleCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    //m_dirCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    //m_mappingCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    //m_centerCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    //m_linedCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    //m_boundaryIndices.reset(new std::vector<int>);
    //m_angleCloudIndices.clear();
    //m_density.clear();
    //m_offsetRate.clear();
    //m_errors.clear();
    //m_linePointsCount.clear();

    //QList<LineSegment> lines;
    //m_lineSegments.clear();

    pcl::PointCloud<pcl::PointXYZI>::Ptr outCloud;
    pcl::IndicesPtr outIndices;
    pcl::PointCloud<pcl::PointXYZI>::Ptr outInlierCloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr outOutlierCloud; 
    pcl::PointCloud<pcl::PointXYZI>::Ptr outAngleCloud;
    QList<int> outAngleCloudIndices;
    pcl::PointCloud<pcl::PointXYZI>::Ptr outDirCloud;
    pcl::PointCloud<pcl::PointXYZI>::Ptr outMappingCloud;
    QList<float> outDensityList;
    QList<float> outOffsetRateList;
    pcl::PointCloud<Line>::Ptr outLineCloud;
    pcl::PointCloud<PointLine>::Ptr outPointLineCloud;
    QMap<int, std::vector<int>> outSubCloudIndices;
    pcl::PointCloud<pcl::PointXYZI>::Ptr outLinedCloud;
    QList<LineSegment> outLineSegments;
    QList<float> outErrors;
    QList<int> outLinePointsCount;

    //pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
    // ��Boundary���Corner��ֱ����Ư�ơ�
    computeInternal(boundaryCloud, 
        BoundaryCloudSearchRadius(), 
        BoundaryCloudA1dThreshold(),
        BoundaryLineInterval(),
        BoundaryMaxZDistance(),
        outCloud,
        outIndices,
        outInlierCloud,
        outOutlierCloud,
        outAngleCloud,
        outAngleCloudIndices,
        outDirCloud,
        outMappingCloud,
        outDensityList,
        outOffsetRateList,
        outLineCloud,
        outPointLineCloud,
        outSubCloudIndices,
        outLinedCloud,
        outLineSegments,
        outErrors,
        outLinePointsCount
        );
    *m_cloud += *outCloud;
    m_lineSegments += outLineSegments;
    *m_lineCloud += *outLineCloud;

    computeInternal(cornerCloud, 
        CornerCloudSearchRadius(), 
        CornerCloudA1dThreshold(),
        CornerLineInterval(),
        CornerMaxZDistance(),
        outCloud,
        outIndices,
        outInlierCloud,
        outOutlierCloud,
        outAngleCloud,
        outAngleCloudIndices,
        outDirCloud,
        outMappingCloud,
        outDensityList,
        outOffsetRateList,
        outLineCloud,
        outPointLineCloud,
        outSubCloudIndices,
        outLinedCloud,
        outLineSegments,
        outErrors,
        outLinePointsCount
        );
    QList<LineSegment> groupedLineSegments;
    groupLines(outLineSegments, outPointLineCloud, CornerGroupLinesSearchRadius(), outLineCloud, groupedLineSegments);
    *m_cloud += *outCloud;
    m_lineSegments += groupedLineSegments;
    //m_lineSegments += outLineSegments;
    *m_lineCloud += *outLineCloud;

    for (int i = 0; i < m_lineCloud->size(); i++)
    {
        Line& line = m_lineCloud->points[i];

        line.generateDescriptor();
        m_lineCloud->points[i].debugPrint();
    }
    
    m_cloud->width = m_cloud->points.size();
    m_cloud->height = 1;
    m_cloud->is_dense = true;
    /*m_angleCloud->width = m_angleCloud->points.size();
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
    m_centerCloud->is_dense = true;*/
    
    //m_linedCloud->width = m_linedCloud->points.size();
    //m_linedCloud->height = 1;
    //m_linedCloud->is_dense = true;
    //qDebug() << "Extract" << m_subCloudIndices.size() << "groups.";
    //generateDescriptors();
    
    qDebug() << "[LineExtractor::compute] exit";
    return m_lineSegments;
}

void LineExtractor::computeInternal(const pcl::PointCloud<pcl::PointXYZI>::Ptr& inCloud, 
                                    float radius, float a1dThreshold, float lineInterval, float maxZdistance,
                                    pcl::PointCloud<pcl::PointXYZI>::Ptr& outCloud,
                                    pcl::IndicesPtr& outIndices,
                                    pcl::PointCloud<pcl::PointXYZI>::Ptr& outInlierCloud,
                                    pcl::PointCloud<pcl::PointXYZI>::Ptr& outOutlierCloud, 
                                    pcl::PointCloud<pcl::PointXYZI>::Ptr& outAngleCloud,
                                    QList<int>& outAngleCloudIndices,
                                    pcl::PointCloud<pcl::PointXYZI>::Ptr& outDirCloud,
                                    pcl::PointCloud<pcl::PointXYZI>::Ptr& outMappingCloud,
                                    QList<float>& outDensityList,
                                    QList<float>& outOffsetRateList,
                                    pcl::PointCloud<Line>::Ptr& outLineCloud,
                                    pcl::PointCloud<PointLine>::Ptr& outPointLineCloud,
                                    QMap<int, std::vector<int>>& outSubCloudIndices,
                                    pcl::PointCloud<pcl::PointXYZI>::Ptr& outLinedCloud,
                                    QList<LineSegment>& outLineSegments,
                                    QList<float>& outErrors,
                                    QList<int>& outLinePointsCount)
{
    outInlierCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    outOutlierCloud.reset(new pcl::PointCloud<pcl::PointXYZI>); 
    outAngleCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    outAngleCloudIndices = QList<int>();
    outDirCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    outMappingCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    outDensityList = QList<float>();
    outOffsetRateList = QList<float>();
    outLineCloud.reset(new pcl::PointCloud<Line>);
    outPointLineCloud.reset(new pcl::PointCloud<PointLine>);
    outSubCloudIndices = QMap<int, std::vector<int>>();
    outLinedCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    outLineSegments = QList<LineSegment>();
    outErrors = QList<float>();
    outLinePointsCount = QList<int>();
    outCloud.reset(new pcl::PointCloud<pcl::PointXYZI>);
    outIndices.reset(new pcl::Indices);

    // ����ÿһ�����PCA������
    pcl::PointCloud<pcl::PointXYZI>::Ptr tmpCloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::copyPointCloud(*inCloud, *tmpCloud);
    pcl::search::KdTree<pcl::PointXYZI> tree;
    tree.setInputCloud(tmpCloud);

    int index = 0;
    for (int i = 0; i < tmpCloud->size(); i++)
    {
        pcl::PointXYZI ptIn = tmpCloud->points[i];
        Eigen::Vector3f point = ptIn.getVector3fMap();

        // ���ҽ���
        std::vector<int> neighbourIndices;
        std::vector<float> neighbourDistances;
        tree.radiusSearch(ptIn, radius, neighbourIndices, neighbourDistances);

        // ���ڵ�̫�ٱ�ʾ����һ����Ⱥ�㣬��С��3Ҳ�޷�����PCA����
        if (neighbourIndices.size() < MinNeighboursCount())
        {
            continue;
        }

        // PCA���㵱ǰ���������
        pcl::PCA<pcl::PointXYZI> pca;
        pca.setInputCloud(tmpCloud);
        pca.setIndices(pcl::IndicesPtr(new std::vector<int>(neighbourIndices)));
        Eigen::Vector3f eigenValues = pca.getEigenValues();
        Eigen::Vector3f::Index maxIndex;
        eigenValues.maxCoeff(&maxIndex);

        // ������
        Eigen::Vector3f primeDir = pca.getEigenVectors().col(maxIndex).normalized();
        // �е�
        Eigen::Vector3f meanPoint = pca.getMean().head(3);

        Eigen::Vector3f tmpPoint = closedPointOnLine(point, primeDir, meanPoint);

        // ƽ�Ƶ㵽pca���������е�ȷ����ֱ����
        pcl::PointXYZI outPt;
        outPt.getArray3fMap() = tmpPoint;

        if (sqrt(eigenValues[0]) / sqrt(eigenValues[1]) < a1dThreshold)
        {
            // ������ķŵ���㼯��
            outOutlierCloud->points.push_back(outPt);
        }
        else
        {
            //// ������������ƣ���Ҫ��Ϊ�˱��������򣬹���������ʹ�á�
            //pcl::PointXYZI dirPt;
            //dirPt.getVector3fMap() = primeDir;
            //dirPt.intensity = index;
            //outDirCloud->push_back(dirPt);

            // ���С�ķŵ��ڵ㼯��
            outPt.intensity = index;
            outInlierCloud->points.push_back(outPt);
            index++;
        }
    }

    pcl::search::KdTree<pcl::PointXYZI> inlierTree;
    inlierTree.setInputCloud(outInlierCloud);
    //tmpCloud->clear();
    //for (int i = 0; i < outOutlierCloud->points.size(); i++)
    //{
    //    pcl::PointXYZI ptIn = outOutlierCloud->points[i];
    //    Eigen::Vector3f point = ptIn.getVector3fMap();
    //
    //    // ���ҽ���
    //    std::vector<int> neighbourIndices;
    //    std::vector<float> neighbourDistances;
    //    if (!inlierTree.nearestKSearch(ptIn, 1, neighbourIndices, neighbourDistances))
    //        continue;
    //    
    //    pcl::PointXYZI neighbour = outInlierCloud->points[neighbourIndices[0]];
    //    int neighbourIndex = static_cast<int>(neighbour.intensity);
    //    pcl::PointXYZI dirPt = outDirCloud->points[neighbourIndex];

    //    // ƽ�Ƶ㵽pca���������е�ȷ����ֱ����
    //    Eigen::Vector3f tmpPoint = closedPointOnLine(point, dirPt.getVector3fMap(), ptIn.getVector3fMap());
    //    pcl::PointXYZI outPt;
    //    outPt.getArray3fMap() = tmpPoint;

    //    outPt.intensity = index;
    //    dirPt.intensity = index;
    //    tmpCloud->points.push_back(outPt);
    //    outDirCloud->points.push_back(dirPt);
    //    index++;
    //}
    //*outInlierCloud += *tmpCloud;

    // ���㵱ǰ��������е����ֱ������ֵ�����ڶ�ֱ�߳��Ƚ��й�һ���ķ�ĸ��
    Eigen::Vector4f minPoint4, maxPoint4;
    pcl::getMinMax3D<pcl::PointXYZI>(*inCloud, minPoint4, maxPoint4);
    Eigen::Vector3f minPoint = minPoint4.head(3);
    Eigen::Vector3f maxPoint = maxPoint4.head(3);
    float boundBoxDiameter = (maxPoint - minPoint).norm();
    qDebug() << "bound box diameter:" << boundBoxDiameter << minPoint.x() << minPoint.y() << minPoint.z() << maxPoint.x() << maxPoint.y() << maxPoint.z() << (maxPoint - minPoint).norm();

    // globalDir����ͳһ���е�ֱ�߷���
    Eigen::Vector3f globalDir(1, 1, 1);
    globalDir.normalize();
    Eigen::Vector3f xAxis(1, 0, 0);
    Eigen::Vector3f yAxis(0, 1, 0);
    Eigen::Vector3f zAxis(0, 0, 1);

    // ����ÿһ���߽���PCA������
    index = 0;
    for (int i = 0; i < outInlierCloud->size(); i++)
    {
        pcl::PointXYZI ptIn = outInlierCloud->points[i];
        Eigen::Vector3f point = ptIn.getVector3fMap();
        int ptIndex = static_cast<int>(ptIn.intensity);

        // ���ҽ���
        std::vector<int> neighbourIndices;
        std::vector<float> neighbourDistances;
        inlierTree.radiusSearch(ptIn, PCASearchRadius(), neighbourIndices, neighbourDistances);

        // ���ڵ�̫�ٱ�ʾ����һ����Ⱥ�㣬��С��3Ҳ�޷�����PCA����
        if (neighbourIndices.size() < MinNeighboursCount())
        {
            continue;
        }

        // PCA���㵱ǰ���������
        pcl::PCA<pcl::PointXYZI> pca;
        pca.setInputCloud(outInlierCloud);
        pca.setIndices(pcl::IndicesPtr(new std::vector<int>(neighbourIndices)));
        Eigen::Vector3f eigenValues = pca.getEigenValues();
        Eigen::Vector3f::Index maxIndex;
        eigenValues.maxCoeff(&maxIndex);

        // ������
        Eigen::Vector3f primeDir = pca.getEigenVectors().col(maxIndex).normalized();
        // �е�
        Eigen::Vector3f meanPoint = pca.getMean().head(3);

        //Eigen::Vector3f tmpPoint = closedPointOnLine(point, primeDir, meanPoint);

        // ������
        //Eigen::Vector3f primeDir = outInlierCloud->points[ptIndex].getArray3fMap();
        // �е�
        //Eigen::Vector3f meanPoint = ptIn.getArray3fMap();

        //Eigen::Vector3f tmpPoint = closedPointOnLine(point, primeDir, meanPoint);

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

        if (qIsNaN(eulerAngles.x()) || qIsNaN(eulerAngles.y()) || qIsNaN(eulerAngles.z()))
            continue;

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
        outAngleCloud->push_back(anglePt);
        
        // ������������ƣ���Ҫ��Ϊ�˱��������򣬹���������ʹ�á�
        pcl::PointXYZI dirPt;
        dirPt.getVector3fMap() = primeDir;
        dirPt.intensity = index;
        outDirCloud->push_back(dirPt);

        // ��������ԭ�㴹�ࡢ��������нǺ�������ά���������������γ�һ���µ�ӳ����ƣ�����ʹ�þۼ���ȡ������ֱ�ߡ�
        pcl::PointXYZI mappingPt;
        mappingPt.x = distance / boundBoxDiameter * 2;
        mappingPt.y = radiansZ / (M_PI * 2);
        mappingPt.z = xCoord / boundBoxDiameter * 2;
        mappingPt.intensity = index;
        outMappingCloud->push_back(mappingPt);

        // ������������ĵ��������һ�����ƣ�����������㡣
        //pcl::PointXYZI centerPt;
        //centerPt.getVector3fMap() = meanPoint;
        //centerPt.intensity = index;
        //outCenterCloud->push_back(centerPt);

        // ��������ĵ����ܶȵ��������һ�����ƣ�����������㡣
        pcl::PointXYZI pt = ptIn;
        pt.intensity = index;
        outIndices->push_back(index);
        outCloud->push_back(pt);

        outAngleCloudIndices.append(index);
        index++;
    }

    qDebug() << "Input cloud size:" << inCloud->size() << ", filtered cloud size:" << outCloud->size();

    // �������������ƵĲ���������������ͨ���Ƕ�ֵ������������������ͬ�ľۼ�
    pcl::search::KdTree<pcl::PointXYZI> angleCloudTree;
    angleCloudTree.setInputCloud(outAngleCloud);

    // �������ѭ�����ڲ�����������ͬ�ĵ㣬������ʹ���������������������ϴ��ƫ��
    //float maxDensity = 0;
    //float minDensity = m_angleCloud->size();
    QMap<int, std::vector<int>> neighbours;
    for (int i = 0; i < outAngleCloud->size(); i++)
    {
        pcl::PointXYZI ptAngle = outAngleCloud->points[i];

        // ���ҽ���
        std::vector<int> neighbourIndices;
        std::vector<float> neighbourDistances;
        if (!angleCloudTree.radiusSearch(ptAngle, qDegreesToRadians(AngleCloudSearchRadius()) * M_1_PI, neighbourIndices, neighbourDistances))
            continue;

        neighbours.insert(i, neighbourIndices);
        
        // �����ܶ�ֵ
        float density = neighbourIndices.size();

        outDensityList.append(density);

        // ��������ֵ
        Eigen::Vector3f center(Eigen::Vector3f::Zero());
        for (int n = 0; n < neighbourIndices.size(); n++)
        {
            int neighbourIndex = neighbourIndices[n];
            pcl::PointXYZI ptNeighbour = outAngleCloud->points[neighbourIndex];
            center += ptNeighbour.getVector3fMap();
        }
        center /= density;

        // ����������
        float offsetRate = (ptAngle.getVector3fMap() - center).norm();
        outOffsetRateList.append(offsetRate);
    }

    // �����ܶ�ֵ��������ֵ�����е�ӳ����������
    qSort(outAngleCloudIndices.begin(), outAngleCloudIndices.end(), [=](int v1, int v2) -> bool
        {
            if (v1 >= outAngleCloudIndices.size() || v2 >= outAngleCloudIndices.size())
                return false;

            if (outDensityList[v1] == outDensityList[v2])
            {
                return outOffsetRateList[v1] < outOffsetRateList[v2];
            }
            else
            {
                return outDensityList[v1] > outDensityList[v2];
            }
        }
    );
    
    //m_mslCloud.reset(new pcl::PointCloud<Line>);

    // ���ܶ�����ֵ��ʼ����ÿһ��ӳ��㣬ÿһ��ӳ��㼰����ڵ㼴Ϊһ��ͬ����ĵ㣬����Щ��ķ���
    // ���ܽ�����ƽ�ж�����Զ��
    //m_lineCloud.reset(new pcl::PointCloud<PointLine>);
    QVector<bool> processed(outAngleCloud->size(), false);
    for (int i = 0; i < outAngleCloudIndices.size(); i++)
    {
        int index = outAngleCloudIndices[i];
        if (processed[index])
        {
            continue;
        }
        //qDebug() << indexList[i] << m_density[indexList[i]];

        pcl::PointXYZI ptAngle = outAngleCloud->points[index];
        std::vector<int> neighbourIndices = neighbours[index];
        if (neighbourIndices.size() < AngleCloudMinNeighboursCount())
            continue;

        std::vector<int> subCloudIndices;
        for (int n = 0; n < neighbourIndices.size(); n++)
        {
            int neighbourIndex = neighbourIndices[n];
            if (processed[neighbourIndex])
            {
                continue;
            }
            subCloudIndices.push_back(static_cast<int>(outAngleCloud->points[neighbourIndex].intensity));
            processed[neighbourIndex] = true;
        }

        // �����ǰӳ���Ľ��������㹻�����ٴ�ʹ��������ŷʽ�ۼ�����������ٴξۼ������ҳ��������������߶Ρ�
        if (subCloudIndices.size() < AngleCloudMinNeighboursCount())
            continue;

        outSubCloudIndices.insert(outAngleCloud->points[index].intensity, subCloudIndices);

        qDebug() << "index:" << index << ", sub cloud size:" << subCloudIndices.size();

        pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
        tree->setInputCloud(outMappingCloud);

        std::vector<pcl::PointIndices> clusterIndices;

        int count = 0;
        Eigen::Vector3f cecDir(Eigen::Vector3f::Zero());
        // ʹ��������ŷʽ�ۼ������������������ص���x�������������������yz�����������
        pcl::ConditionalEuclideanClustering<pcl::PointXYZI> cec(true);
        cec.setConditionFunction([=](const pcl::PointXYZI& ptSeed, const pcl::PointXYZI& ptCandidate, float sqrDistance) -> bool
            {
                if (qSqrt(sqrDistance) < lineInterval)
                {
                    Eigen::Vector3f seedPoint = ptSeed.getVector3fMap();
                    Eigen::Vector3f candidatePoint = ptCandidate.getVector3fMap();

                    Eigen::Vector3f dir = outDirCloud->points[outAngleCloud->points[index].intensity].getVector3fMap();
                    float distToZ = (candidatePoint - seedPoint).cross(dir).norm();
                    if (distToZ < maxZdistance)
                    {
                        return true;
                    }
                    //return true;
                }
                return false;
            }
        );
        cec.setClusterTolerance(lineInterval);
        cec.setMinClusterSize(AngleCloudMinNeighboursCount());
        cec.setMaxClusterSize(subCloudIndices.size());
        cec.setSearchMethod(tree);
        cec.setInputCloud(outMappingCloud);
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
            if (indices.indices.size() < AngleCloudMinNeighboursCount())
                continue;
            // ����ȫ����ڵ㷽���ƽ���ķ�����Ϊ������ͬʱ�����ʵ㡣
            for (std::vector<int>::const_iterator itIndices = indices.indices.begin(); itIndices != indices.indices.end(); ++itIndices)
            {
                // ȡ��ÿһ���������ֵ
                int localIndex = outMappingCloud->points[*itIndices].intensity;
                // ȡ�����Ӧ��ʵ����ά��
                pcl::PointXYZI ptBoundary = outCloud->points[localIndex];
                // ȡ��֮ǰ���㲢����ĸõ�PCA������������ע��ֻ����һ�����������
                pcl::PointXYZI ptDir = outDirCloud->points[localIndex];

                // ��dir�����ۼ�
                dir += ptDir.getVector3fMap();
                // �����������ۼ�
                center += ptBoundary.getVector3fMap();

                ptBoundary.intensity = index;
                outLinedCloud->push_back(ptBoundary);
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
                int index = outMappingCloud->points[*itIndices].intensity;
                pcl::PointXYZI& ptBoundary = outCloud->points[index];
                ptBoundary.intensity = outPointLineCloud->points.size();
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
            LineSegment ls(start, end);

            qDebug() << "    cluster size:" << indices.indices.size() << ", error:" << error << ", length:" << ls.length();

            if (ls.length() > MinLineLength()/* && error <= 200*/)
            {
                outLineSegments.append(ls);
                outErrors.append(error);
                outLinePointsCount.append(indices.indices.size());

                PointLine pl;
                Eigen::Vector3f dir = ls.direction().normalized();
                calculateAlphaBeta(dir, pl.dAngleX, pl.dAngleY);
                pl.dist = start.cross(dir).norm() * 10;
                calculateAlphaBeta((start.normalized() - dir).normalized(), pl.vAngleX, pl.vAngleY);
                outPointLineCloud->points.push_back(pl);

                Line line;
                line.dir = dir;
                line.point = ls.middle();
                line.weight = 1;
                outLineCloud->push_back(line);
            }
            itClusters++;
        }
    }
}

//void LineExtractor::extractLinesFromPlanes(const QList<Plane>& planes)
//{
//    //m_mslPointCloud.reset(new pcl::PointCloud<MSLPoint>);
//    //m_mslCloud.reset(new pcl::PointCloud<Line>);
//    if (planes.isEmpty())
//        return;
//
//    Eigen::Vector3f globalDir(1, 1, 1);
//
//    if (!m_mslCloud)
//    {
//        m_mslCloud.reset(new pcl::PointCloud<Line>);
//    }
//    if (!m_lineCloud)
//    {
//        m_lineCloud.reset(new pcl::PointCloud<PointLine>);
//    }
//
//    for (int i = 0; i < planes.size(); i++)
//    {
//        Plane plane1 = planes[i];
//        for (int j = i + 1; j < planes.size(); j++)
//        {
//            //if (i == j)
//                //continue;
//
//            Plane plane2 = planes[j];
//
//            float cos = plane1.dir.dot(plane2.dir);
//            if (qAbs(qRadiansToDegrees(qAcos(cos))) < 30.f)
//                continue;
//
//            Eigen::Vector3f crossLine = plane1.dir.cross(plane2.dir).normalized();
//            if (crossLine.dot(globalDir) < 0)
//                crossLine = -crossLine;
//            if (crossLine.isZero())
//                continue;
//
//            // ���ƽ��1�ϵĵ㵽���ߵĴ��߷���
//            Eigen::Vector3f lineOnPlane1 = plane1.dir.cross(crossLine).normalized();
//            lineOnPlane1.normalize();
//            float dist1 = (plane2.point - plane1.point).dot(plane2.dir) / (lineOnPlane1.dot(plane2.dir));
//            // ���ƽ��1�ϵĵ��ڽ�����ͶӰ�Ľ���
//            Eigen::Vector3f crossPoint1 = plane1.point + lineOnPlane1 * dist1;
//
//            //Eigen::Vector3f lineOnPlane2 = plane2.dir.cross(crossLine).normalized();
//            //lineOnPlane2.normalize();
//            //float dist2 = (plane1.point - plane2.point).dot(plane1.dir) / (lineOnPlane2.dot(plane1.dir));
//            //Eigen::Vector3f crossPoint2 = plane2.point + lineOnPlane2 * dist2;
//
//            //Eigen::Vector3f crossPoint = (crossPoint1 + crossPoint2) / 2;
//            //Eigen::Vector3f crossPoint = crossPoint1;
//            Eigen::Vector3f closedPoint = closedPointOnLine(Eigen::Vector3f::Zero(), crossLine, crossPoint1);
//            
//            //Eigen::Vector3f point = closedPointOnLine(plane1.point, line);
//            //MSLPoint mslPoint;
//            //float alpha, beta;
//            //calculateAlphaBeta(crossLine, alpha, beta);
//            //mslPoint.alpha = alpha / M_PI;
//            //mslPoint.beta = beta / M_PI;
//            //mslPoint.x = closedPoint.x() / m_boundBoxDiameter;
//            //mslPoint.y = closedPoint.y() / m_boundBoxDiameter;
//            //mslPoint.z = closedPoint.z() / m_boundBoxDiameter;
//            //m_mslPointCloud->push_back(mslPoint);
//            Eigen::Vector3f start = closedPoint - crossLine * 0.075f;
//            Eigen::Vector3f end = closedPoint + crossLine * 0.075f;
//            LineSegment ls(start, end);
//            m_lineSegments.append(ls);
//
//            PointLine pl;
//            Eigen::Vector3f dir = ls.direction().normalized();
//            calculateAlphaBeta(dir, pl.dAngleX, pl.dAngleY);
//            pl.dist = start.cross(dir).norm();
//            calculateAlphaBeta((start.normalized() - dir).normalized(), pl.vAngleX, pl.vAngleY);
//            m_lineCloud->points.push_back(pl);
//
//            Line msl;
//            msl.dir = crossLine;
//            msl.point = closedPoint;
//            msl.weight = 1;
//            m_mslCloud->push_back(msl);
//        }
//    }
//
//    //m_mslPointCloud->width = m_mslPointCloud->points.size();
//    //m_mslPointCloud->height = 1;
//    //m_mslPointCloud->is_dense = true;
//}

void LineExtractor::groupLines(
    const QList<LineSegment>& inLineSegments,
    const pcl::PointCloud<PointLine>::Ptr& inLineCloud,
    float groupLinesSearchRadius,
    pcl::PointCloud<Line>::Ptr& outLineCloud,
    QList<LineSegment>& outLineSegments
)
{
    outLineCloud.reset(new pcl::PointCloud<Line>);
    outLineSegments = QList<LineSegment>();
    if (inLineSegments.size() > 0)
    {
        pcl::KdTreeFLANN<PointLine> lineTree;
        lineTree.setInputCloud(inLineCloud);

        QMap<int, bool> lineProcessed;
        QVector<float> lineDense(inLineSegments.size());
        QVector<int> lineIndices(inLineSegments.size());
        for (int i = 0; i < inLineSegments.size(); i++)
        {
            std::vector<int> indices;
            std::vector<float> dists;
            lineTree.radiusSearch(inLineCloud->points[i], groupLinesSearchRadius, indices, dists);
            lineDense[i] = indices.size();
            lineIndices[i] = i;

            qDebug() << "line" << i << ", dense:" << indices.size() << ", dist:" << qSqrt(dists[0]);
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

            PointLine pl = inLineCloud->points[index];
            std::vector<int> indices;
            std::vector<float> dists;
            lineTree.radiusSearch(inLineCloud->points[index], groupLinesSearchRadius, indices, dists);
            qDebug() << "line" << index << ", merging lines:" << indices.size();
            Eigen::Vector3f dir(Eigen::Vector3f::Zero());
            Eigen::Vector3f point(Eigen::Vector3f::Zero());
            Eigen::Vector3f start(Eigen::Vector3f::Zero());
            Eigen::Vector3f end(Eigen::Vector3f::Zero());
            float sumLength = 0;
            for (int i = 0; i < indices.size(); i++)
            {
                int neighbourIndex = indices[i];
                qDebug() << "    " << neighbourIndex;
                lineProcessed[neighbourIndex] = true;
                LineSegment ls = inLineSegments[neighbourIndex];
                sumLength += ls.length();

                if (point.isZero())
                {
                    point = ls.middle();
                    start = ls.start();
                    end = ls.end();
                }
                else
                {
                    point = (point * i + ls.middle()) / (i + 1);
                    if ((ls.start() - point).squaredNorm() > (start - point).squaredNorm()/* && (ls.start() - point).dot(start - point) > 0*/)
                    {
                        start = ls.start();
                    }
                    //if ((ls.end() - point).squaredNorm() > (start - point).squaredNorm() && (ls.end() - point).dot(start - point) > 0)
                    //{
                        //start = ls.end();
                    //}
                    //if ((ls.start() - point).squaredNorm() > (end - point).squaredNorm()/* && (ls.start() - point).dot(end - point) > 0*/)
                    //{
                        //end = ls.start();
                    //}
                    if ((ls.end() - point).squaredNorm() > (end - point).squaredNorm()/* && (ls.end() - point).dot(end - point) > 0*/)
                    {
                        end = ls.end();
                    }
                }
            }
            //point = Eigen::Vector3f::Zero();
            for (int i = 0; i < indices.size(); i++)
            {
                int neighbourIndex = indices[i];
                LineSegment ls = inLineSegments[neighbourIndex];
                dir += ls.direction().normalized() * ls.length() / sumLength;
                //point += ls.middle() * ls.length() / sumLength;
            }
            dir.normalize();

            start = closedPointOnLine(start, dir, point);
            end = closedPointOnLine(end, dir, point);
            LineSegment ls(start, end, i);
            Line line;
            line.dir = dir;
            line.point = point + dir * (point.dot(dir));
            line.weight = 1;
            //line.generateDescriptor();
            //line.debugPrint();
            outLineCloud->push_back(line);
            outLineSegments.append(ls);
        }
        outLineCloud->width = outLineCloud->points.size();
        outLineCloud->height = 1;
        outLineCloud->is_dense = true;
    }
}

//void LineExtractor::generateDescriptors()
//{
//    m_descriptors.reset(new pcl::PointCloud<LineDescriptor>);
//    for (int i = 0; i < m_lineCloud->size(); i++)
//    {
//        Line& line = m_lineCloud->points[i];
//
//        line.generateDescriptor();
//        //m_descriptors->points.push_back(desc);
//    }
//}

