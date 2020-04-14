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
    // 对Boundary点和Corner点分别进行漂移。
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

    // 计算每一个点的PCA主方向，
    pcl::PointCloud<pcl::PointXYZI>::Ptr tmpCloud(new pcl::PointCloud<pcl::PointXYZI>);
    pcl::copyPointCloud(*inCloud, *tmpCloud);
    pcl::search::KdTree<pcl::PointXYZI> tree;
    tree.setInputCloud(tmpCloud);

    int index = 0;
    for (int i = 0; i < tmpCloud->size(); i++)
    {
        pcl::PointXYZI ptIn = tmpCloud->points[i];
        Eigen::Vector3f point = ptIn.getVector3fMap();

        // 查找近邻
        std::vector<int> neighbourIndices;
        std::vector<float> neighbourDistances;
        tree.radiusSearch(ptIn, radius, neighbourIndices, neighbourDistances);

        // 近邻点太少表示这是一个离群点，且小于3也无法进行PCA计算
        if (neighbourIndices.size() < MinNeighboursCount())
        {
            continue;
        }

        // PCA计算当前点的主方向
        pcl::PCA<pcl::PointXYZI> pca;
        pca.setInputCloud(tmpCloud);
        pca.setIndices(pcl::IndicesPtr(new std::vector<int>(neighbourIndices)));
        Eigen::Vector3f eigenValues = pca.getEigenValues();
        Eigen::Vector3f::Index maxIndex;
        eigenValues.maxCoeff(&maxIndex);

        // 主方向
        Eigen::Vector3f primeDir = pca.getEigenVectors().col(maxIndex).normalized();
        // 中点
        Eigen::Vector3f meanPoint = pca.getMean().head(3);

        Eigen::Vector3f tmpPoint = closedPointOnLine(point, primeDir, meanPoint);

        // 平移点到pca主方向与中点确定的直线上
        pcl::PointXYZI outPt;
        outPt.getArray3fMap() = tmpPoint;

        if (sqrt(eigenValues[0]) / sqrt(eigenValues[1]) < a1dThreshold)
        {
            // 误差过大的放到外点集中
            outOutlierCloud->points.push_back(outPt);
        }
        else
        {
            //// 建立主方向点云，主要是为了保留主方向，供后续计算使用。
            //pcl::PointXYZI dirPt;
            //dirPt.getVector3fMap() = primeDir;
            //dirPt.intensity = index;
            //outDirCloud->push_back(dirPt);

            // 误差小的放到内点集中
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
    //    // 查找近邻
    //    std::vector<int> neighbourIndices;
    //    std::vector<float> neighbourDistances;
    //    if (!inlierTree.nearestKSearch(ptIn, 1, neighbourIndices, neighbourDistances))
    //        continue;
    //    
    //    pcl::PointXYZI neighbour = outInlierCloud->points[neighbourIndices[0]];
    //    int neighbourIndex = static_cast<int>(neighbour.intensity);
    //    pcl::PointXYZI dirPt = outDirCloud->points[neighbourIndex];

    //    // 平移点到pca主方向与中点确定的直线上
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

    // 计算当前点云外包盒的最大直径，该值将用于对直线长度进行归一化的分母。
    Eigen::Vector4f minPoint4, maxPoint4;
    pcl::getMinMax3D<pcl::PointXYZI>(*inCloud, minPoint4, maxPoint4);
    Eigen::Vector3f minPoint = minPoint4.head(3);
    Eigen::Vector3f maxPoint = maxPoint4.head(3);
    float boundBoxDiameter = (maxPoint - minPoint).norm();
    qDebug() << "bound box diameter:" << boundBoxDiameter << minPoint.x() << minPoint.y() << minPoint.z() << maxPoint.x() << maxPoint.y() << maxPoint.z() << (maxPoint - minPoint).norm();

    // globalDir用于统一所有的直线方向
    Eigen::Vector3f globalDir(1, 1, 1);
    globalDir.normalize();
    Eigen::Vector3f xAxis(1, 0, 0);
    Eigen::Vector3f yAxis(0, 1, 0);
    Eigen::Vector3f zAxis(0, 0, 1);

    // 计算每一个边界点的PCA主方向，
    index = 0;
    for (int i = 0; i < outInlierCloud->size(); i++)
    {
        pcl::PointXYZI ptIn = outInlierCloud->points[i];
        Eigen::Vector3f point = ptIn.getVector3fMap();
        int ptIndex = static_cast<int>(ptIn.intensity);

        // 查找近邻
        std::vector<int> neighbourIndices;
        std::vector<float> neighbourDistances;
        inlierTree.radiusSearch(ptIn, PCASearchRadius(), neighbourIndices, neighbourDistances);

        // 近邻点太少表示这是一个离群点，且小于3也无法进行PCA计算
        if (neighbourIndices.size() < MinNeighboursCount())
        {
            continue;
        }

        // PCA计算当前点的主方向
        pcl::PCA<pcl::PointXYZI> pca;
        pca.setInputCloud(outInlierCloud);
        pca.setIndices(pcl::IndicesPtr(new std::vector<int>(neighbourIndices)));
        Eigen::Vector3f eigenValues = pca.getEigenValues();
        Eigen::Vector3f::Index maxIndex;
        eigenValues.maxCoeff(&maxIndex);

        // 主方向
        Eigen::Vector3f primeDir = pca.getEigenVectors().col(maxIndex).normalized();
        // 中点
        Eigen::Vector3f meanPoint = pca.getMean().head(3);

        //Eigen::Vector3f tmpPoint = closedPointOnLine(point, primeDir, meanPoint);

        // 主方向
        //Eigen::Vector3f primeDir = outInlierCloud->points[ptIndex].getArray3fMap();
        // 中点
        //Eigen::Vector3f meanPoint = ptIn.getArray3fMap();

        //Eigen::Vector3f tmpPoint = closedPointOnLine(point, primeDir, meanPoint);

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

        if (qIsNaN(eulerAngles.x()) || qIsNaN(eulerAngles.y()) || qIsNaN(eulerAngles.z()))
            continue;

        // 计算当前点到穿过原点的主方向所在直线的垂直距离
        float distance = point.cross(primeDir).norm();

        // 计算当前点投影到穿过原点的主方向所在直线的投影点的有向长度
        Eigen::Vector3f pointProjToPrimeDir = primeDir * point.dot(primeDir);
        float xCoord = pointProjToPrimeDir.norm();
        if (pointProjToPrimeDir.dot(primeDir) < 0)
        {
            xCoord = -xCoord;
        }

        // 当前点与投影到穿过原点的主方向直线的投影点所形成的直线，计算该直线与深度方向叉乘pca主方向的角度
        Eigen::Vector3f lineVertProj = (point - pointProjToPrimeDir).normalized();
        float cosZ = lineVertProj.dot(zAxis.cross(primeDir).normalized());
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
        outAngleCloud->push_back(anglePt);
        
        // 建立主方向点云，主要是为了保留主方向，供后续计算使用。
        pcl::PointXYZI dirPt;
        dirPt.getVector3fMap() = primeDir;
        dirPt.intensity = index;
        outDirCloud->push_back(dirPt);

        // 用主方向原点垂距、主方向垂面夹角和主方向单维距离这三个数据形成一个新的映射点云，用于使用聚集抽取连续的直线。
        pcl::PointXYZI mappingPt;
        mappingPt.x = distance / boundBoxDiameter * 2;
        mappingPt.y = radiansZ / (M_PI * 2);
        mappingPt.z = xCoord / boundBoxDiameter * 2;
        mappingPt.intensity = index;
        outMappingCloud->push_back(mappingPt);

        // 将计算出的质心单独保存成一个点云，方便后续计算。
        //pcl::PointXYZI centerPt;
        //centerPt.getVector3fMap() = meanPoint;
        //centerPt.intensity = index;
        //outCenterCloud->push_back(centerPt);

        // 将计算出的点云密度单独保存成一个点云，方便后续计算。
        pcl::PointXYZI pt = ptIn;
        pt.intensity = index;
        outIndices->push_back(index);
        outCloud->push_back(pt);

        outAngleCloudIndices.append(index);
        index++;
    }

    qDebug() << "Input cloud size:" << inCloud->size() << ", filtered cloud size:" << outCloud->size();

    // 创建参数化点云的查找树，该树用于通过角度值点云来查找主方向相同的聚集
    pcl::search::KdTree<pcl::PointXYZI> angleCloudTree;
    angleCloudTree.setInputCloud(outAngleCloud);

    // 下面这个循环用于查找主方向相同的点，但并不使用区域增长，否则会产生较大的偏离
    //float maxDensity = 0;
    //float minDensity = m_angleCloud->size();
    QMap<int, std::vector<int>> neighbours;
    for (int i = 0; i < outAngleCloud->size(); i++)
    {
        pcl::PointXYZI ptAngle = outAngleCloud->points[i];

        // 查找近邻
        std::vector<int> neighbourIndices;
        std::vector<float> neighbourDistances;
        if (!angleCloudTree.radiusSearch(ptAngle, qDegreesToRadians(AngleCloudSearchRadius()) * M_1_PI, neighbourIndices, neighbourDistances))
            continue;

        neighbours.insert(i, neighbourIndices);
        
        // 计算密度值
        float density = neighbourIndices.size();

        outDensityList.append(density);

        // 计算质心值
        Eigen::Vector3f center(Eigen::Vector3f::Zero());
        for (int n = 0; n < neighbourIndices.size(); n++)
        {
            int neighbourIndex = neighbourIndices[n];
            pcl::PointXYZI ptNeighbour = outAngleCloud->points[neighbourIndex];
            center += ptNeighbour.getVector3fMap();
        }
        center /= density;

        // 计算离心率
        float offsetRate = (ptAngle.getVector3fMap() - center).norm();
        outOffsetRateList.append(offsetRate);
    }

    // 根据密度值和离心率值对所有的映射点进行排序。
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

    // 从密度最大的值开始挑出每一个映射点，每一个映射点及其近邻点即为一组同方向的点，但这些点的方向
    // 可能仅仅是平行而相距很远。
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

        // 如果当前映射点的近邻数量足够，则再次使用条件化欧式聚集来对这组点再次聚集，以找出共线且连续的线段。
        if (subCloudIndices.size() < AngleCloudMinNeighboursCount())
            continue;

        outSubCloudIndices.insert(outAngleCloud->points[index].intensity, subCloudIndices);

        qDebug() << "index:" << index << ", sub cloud size:" << subCloudIndices.size();

        pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
        tree->setInputCloud(outMappingCloud);

        std::vector<pcl::PointIndices> clusterIndices;

        int count = 0;
        Eigen::Vector3f cecDir(Eigen::Vector3f::Zero());
        // 使用条件化欧式聚集来进行区域增长，重点在x方向的增长，而限制在yz方向的增长。
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
            // 将分割出的聚集抽取成直线
            pcl::PointIndices indices = *itClusters;
            Eigen::Vector3f dir(0, 0, 0);
            Eigen::Vector3f start(0, 0, 0);
            Eigen::Vector3f end(0, 0, 0);
            Eigen::Vector3f center(0, 0, 0);

            // 判断组成聚集的点数是否满足要求。
            if (indices.indices.size() < AngleCloudMinNeighboursCount())
                continue;
            // 计算全体近邻点方向的平均的方向作为主方向，同时计算质点。
            for (std::vector<int>::const_iterator itIndices = indices.indices.begin(); itIndices != indices.indices.end(); ++itIndices)
            {
                // 取出每一个点的索引值
                int localIndex = outMappingCloud->points[*itIndices].intensity;
                // 取出其对应的实际三维点
                pcl::PointXYZI ptBoundary = outCloud->points[localIndex];
                // 取出之前计算并保存的该点PCA计算后的主方向，注意只是这一个点的主方向
                pcl::PointXYZI ptDir = outDirCloud->points[localIndex];

                // 向dir变量累加
                dir += ptDir.getVector3fMap();
                // 向质量变量累加
                center += ptBoundary.getVector3fMap();

                ptBoundary.intensity = index;
                outLinedCloud->push_back(ptBoundary);
            }
            // 求均值后，得到当前这个聚集的平均方向和质心
            dir /= indices.indices.size();
            dir.normalized();
            center /= indices.indices.size();

            // 验证直线，计算每个点到目标直线的距离期望，使用该值作为误差以验证是否为有效的直线。
            // 同时计算这一组点形成的起点与终点。
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
//            // 求得平面1上的点到交线的垂线方向
//            Eigen::Vector3f lineOnPlane1 = plane1.dir.cross(crossLine).normalized();
//            lineOnPlane1.normalize();
//            float dist1 = (plane2.point - plane1.point).dot(plane2.dir) / (lineOnPlane1.dot(plane2.dir));
//            // 求得平面1上的点在交线上投影的交点
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

